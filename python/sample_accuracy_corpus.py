#!/usr/bin/env python3
"""Build a fixed, reproducible accuracy-regression corpus from KataGo training shards.

Unlike build_accuracy_dataset.py, this sampler keeps the raw packed features exactly as
written by the C++ selfplay/training writer (fillRowV7 output).  History is already encoded
in the spatial (channels 9-13) and global (channels 0-4) features of each row, so no
history-matrix transform is applied and replay never re-randomizes history or symmetry.

The output .npz stores:
  binaryInputNCHWPacked  (N, 22, 46)  uint8    raw packed 19x19 spatial features
  globalInputNC          (N, 19)      float32  raw global/rule features
  policyTargetsNCMove    (N, 2, 362)  float32  player & opponent policy target counts
  globalTargetsNC        (N, 80)      float32  training targets incl. weights
  scoreDistrN            (N, 842)     float32
  valueTargetsNCHW       (N, 5, 19, 19) float32
  qValueTargetsNCMove    (N, 3, 362)  float32

Sampling is uniform without replacement over the global row index space; each row is
equally likely regardless of shard size.

Usage:
  python sample_accuracy_corpus.py \
    --input-dir /workspace/trainingdata/extracted/2026-08-01npzs \
    --source-archive /workspace/trainingdata/2026-08-01npzs.tgz \
    --pos-len 19 --num-samples 8192 --seed 20260803 \
    --output-npz /workspace/trainingdata/accuracy/2026-08-01-19x19-8192-seed20260803.npz \
    --manifest-json /workspace/trainingdata/accuracy/2026-08-01-19x19-8192-seed20260803.manifest.json
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

GLOBAL_TARGETS_NC_CHANNELS = 80
POS_LEN = 19
NUM_BIN_FEATURES = 22
NUM_GLOBAL_FEATURES = 19
PACKED_WIDTH = (POS_LEN * POS_LEN + 7) // 8
POLICY_LEN = POS_LEN * POS_LEN + 1


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pad_global_targets_nc(global_targets_nc):
    num_channels = global_targets_nc.shape[1]
    if num_channels == GLOBAL_TARGETS_NC_CHANNELS:
        return global_targets_nc
    if num_channels > GLOBAL_TARGETS_NC_CHANNELS:
        raise ValueError(
            f"globalTargetsNC has {num_channels} channels, expected at most "
            f"{GLOBAL_TARGETS_NC_CHANNELS}"
        )
    padded = np.zeros(
        (global_targets_nc.shape[0], GLOBAL_TARGETS_NC_CHANNELS),
        dtype=global_targets_nc.dtype,
    )
    padded[:, :num_channels] = global_targets_nc
    return padded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--manifest-json", required=True)
    args = parser.parse_args()

    npz_files = sorted(
        glob.glob(os.path.join(args.input_dir, "**", "*.npz"), recursive=True)
    )
    if not npz_files:
        raise ValueError(f"No .npz files found under {args.input_dir}")

    row_counts = []
    full_row_counts = []
    field_info = {}
    source_hashes = {}
    total_rows = 0
    total_full_rows = 0
    for npz_file in npz_files:
        with np.load(npz_file) as npz:
            for key in npz.files:
                if key not in field_info:
                    field_info[key] = {
                        "shape": [int(x) for x in npz[key].shape[1:]],
                        "dtype": str(npz[key].dtype),
                    }
            packed_shape = npz["binaryInputNCHWPacked"].shape
            if packed_shape[1] != NUM_BIN_FEATURES or packed_shape[2] != PACKED_WIDTH:
                raise ValueError(
                    f"{npz_file}: unexpected binaryInputNCHWPacked shape {packed_shape}"
                )
            if npz["globalInputNC"].shape[1] != NUM_GLOBAL_FEATURES:
                raise ValueError(
                    f"{npz_file}: unexpected globalInputNC shape {npz['globalInputNC'].shape}"
                )
            n = int(packed_shape[0])
            row_counts.append(n)
            total_rows += n
            # Keep only full pos_len x pos_len rows: channel 0 (the on-board mask) must be all
            # ones after unpacking. Smaller boards (e.g. 13x13) are zero-padded in the packed
            # arrays and must be excluded so the corpus is uniform 19x19.
            masks = np.unpackbits(npz["binaryInputNCHWPacked"][:, 0:1, :], axis=2)[:, 0, : POS_LEN * POS_LEN]
            full = int(np.all(masks == 1, axis=1).sum())
            full_row_counts.append(full)
            total_full_rows += full
        source_hashes[os.path.relpath(npz_file, args.input_dir).replace(os.sep, "/")] = (
            sha256_file(npz_file)
        )

    if total_full_rows < args.num_samples:
        raise ValueError(
            f"Only {total_full_rows} full-size 19x19 rows available, requested {args.num_samples}"
        )

    rng = np.random.default_rng(args.seed)
    # Uniform sampling over the global index space of full-size rows only.
    selected_flat = np.sort(rng.choice(total_full_rows, size=args.num_samples, replace=False))

    full_offsets = np.cumsum([0] + full_row_counts)
    shard_groups = {}
    for flat_index in selected_flat:
        shard_idx = int(np.searchsorted(full_offsets[1:], flat_index, side="right"))
        rank_in_shard = int(flat_index - full_offsets[shard_idx])
        shard_groups.setdefault(shard_idx, []).append(rank_in_shard)

    n = args.num_samples
    packed_out = np.zeros((n, NUM_BIN_FEATURES, PACKED_WIDTH), dtype=np.uint8)
    global_out = np.zeros((n, NUM_GLOBAL_FEATURES), dtype=np.float32)
    policy_out = np.zeros((n, 2, POLICY_LEN), dtype=np.float32)
    global_targets_out = np.zeros((n, GLOBAL_TARGETS_NC_CHANNELS), dtype=np.float32)
    score_out = np.zeros((n, 842), dtype=np.float32)
    value_out = np.zeros((n, 5, POS_LEN, POS_LEN), dtype=np.float32)
    qvalue_out = np.zeros((n, 3, POLICY_LEN), dtype=np.float32)

    output_pos = 0
    for shard_idx, rows in sorted(shard_groups.items()):
        rows_arr = np.array(rows, dtype=np.int64)
        with np.load(npz_files[shard_idx]) as npz:
            # ranks in `rows` index into the shard's full-size rows; map back to source rows.
            masks = np.unpackbits(npz["binaryInputNCHWPacked"][:, 0:1, :], axis=2)[:, 0, : POS_LEN * POS_LEN]
            full_row_idxs = np.nonzero(np.all(masks == 1, axis=1))[0]
            source_rows = full_row_idxs[rows_arr]
            end = output_pos + rows_arr.size
            packed_out[output_pos:end] = npz["binaryInputNCHWPacked"][source_rows]
            global_out[output_pos:end] = npz["globalInputNC"][source_rows].astype(np.float32)
            policy_out[output_pos:end] = npz["policyTargetsNCMove"][source_rows].astype(
                np.float32
            )
            global_targets_out[output_pos:end] = pad_global_targets_nc(
                npz["globalTargetsNC"][source_rows]
            ).astype(np.float32)
            score_out[output_pos:end] = npz["scoreDistrN"][source_rows].astype(np.float32)
            value_out[output_pos:end] = npz["valueTargetsNCHW"][source_rows].astype(
                np.float32
            )
            qvalue_out[output_pos:end] = npz["qValueTargetsNCMove"][source_rows].astype(
                np.float32
            )
            group_rows = []
            for rank, src_row in zip(rows, source_rows):
                group_rows.append(int(src_row))
            shard_groups[shard_idx] = group_rows
            output_pos = end

    os.makedirs(os.path.dirname(args.output_npz), exist_ok=True)
    np.savez(
        args.output_npz,
        binaryInputNCHWPacked=packed_out,
        globalInputNC=global_out,
        policyTargetsNCMove=policy_out,
        globalTargetsNC=global_targets_out,
        scoreDistrN=score_out,
        valueTargetsNCHW=value_out,
        qValueTargetsNCMove=qvalue_out,
    )
    output_hash = sha256_file(args.output_npz)
    archive_hash = sha256_file(args.source_archive)

    output_field_info = {}
    with np.load(args.output_npz) as npz:
        output_field_info = {
            key: {
                "shape": [int(x) for x in npz[key].shape],
                "dtype": str(npz[key].dtype),
            }
            for key in npz.files
        }

    shard_list = [
        os.path.relpath(path, args.input_dir).replace(os.sep, "/") for path in npz_files
    ]
    row_manifest = [
        {"shard": shard_list[shard_idx], "row": row}
        for shard_idx, rows in sorted(shard_groups.items())
        for row in rows
    ]

    manifest = {
        "dataset": "accuracy-regression",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_config": "b11c768h12nbt3tflrs-fson-silu (v17, 22 spatial + 19 global features)",
        "pos_len": POS_LEN,
        "num_samples": n,
        "seed": args.seed,
        "sampling": "uniform without replacement over global row index space",
        "features": "raw fillRowV7 output as written by C++ training writer; no history/symmetry transform",
        "source_archive": os.path.basename(args.source_archive),
        "source_archive_sha256": archive_hash,
        "num_shards": len(npz_files),
        "total_rows_seen": total_rows,
        "total_full_19x19_rows": total_full_rows,
        "filter": "keep rows whose on-board mask channel is all ones (full 19x19)",
        "shard_list": shard_list,
        "shard_sha256": source_hashes,
        "rows": row_manifest,
        "source_field_info": field_info,
        "output_npz": os.path.basename(args.output_npz),
        "output_npz_sha256": output_hash,
        "output_field_info": output_field_info,
    }
    with open(args.manifest_json, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {args.output_npz} ({output_hash})")
    print(f"Wrote {args.manifest_json}")
    print(
        f"Rows={n} shards={len(npz_files)} total_rows_seen={total_rows} full_rows={total_full_rows} "
        f"touched_shards={len(shard_groups)}"
    )


if __name__ == "__main__":
    main()
