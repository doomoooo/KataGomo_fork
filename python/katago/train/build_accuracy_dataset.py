#!/usr/bin/env python3
"""Build a fixed, reproducible accuracy-regression dataset from KataGo training shards.

The output stores the network input *after* the official training pipeline's history-matrix
transform, so replay never re-randomizes history or symmetry. Targets needed for all output heads
and p0loss are saved alongside the inputs.

Usage example:
  PYTHONPATH=/workspace/katago/python python -m katago.train.build_accuracy_dataset \
    --input-dir /workspace/trainingdata/extracted/2026-08-01npzs \
    --source-archive /workspace/trainingdata/2026-08-01npzs.tgz \
    --model-config-name b11c768h12nbt3tflrs-fson-silu \
    --pos-len 19 --num-samples 8192 --seed 20260803 --history-seed 20260803 \
    --output-npz /workspace/trainingdata/accuracy/2026-08-01-19x19-8192-seed20260803-p0loss.npz \
    --manifest-json /workspace/trainingdata/accuracy/2026-08-01-19x19-8192-seed20260803-p0loss.manifest.json
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from katago.train import modelconfigs  # noqa: E402

GLOBAL_TARGETS_NC_CHANNELS = 80


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


def build_history_matrices():
    num_bin_features = 22
    h_base = np.diag(
        np.array(
            [
                1.0,  # 0
                1.0,  # 1
                1.0,  # 2
                1.0,  # 3
                1.0,  # 4
                1.0,  # 5
                1.0,  # 6
                1.0,  # 7
                1.0,  # 8
                0.0,  # 9   Location of move 1 turn ago
                0.0,  # 10  Location of move 2 turns ago
                0.0,  # 11  Location of move 3 turns ago
                0.0,  # 12  Location of move 4 turns ago
                0.0,  # 13  Location of move 5 turns ago
                1.0,  # 14  Ladder-threatened stone
                0.0,  # 15  Ladder-threatened stone, 1 turn ago
                0.0,  # 16  Ladder-threatened stone, 2 turns ago
                1.0,  # 17
                1.0,  # 18
                1.0,  # 19
                1.0,  # 20
                1.0,  # 21
            ],
            dtype=np.float32,
        )
    )
    h_base[14, 15] = 1.0
    h_base[14, 16] = 1.0

    def zero_matrix():
        return np.zeros((num_bin_features, num_bin_features), dtype=np.float32)

    h0 = zero_matrix()
    h0[9, 9] = 1.0
    h0[14, 15] = -1.0
    h0[14, 16] = -1.0
    h0[15, 15] = 1.0
    h0[15, 16] = 1.0

    h1 = zero_matrix()
    h1[10, 10] = 1.0
    h1[15, 16] = -1.0
    h1[16, 16] = 1.0

    h2 = zero_matrix()
    h2[11, 11] = 1.0

    h3 = zero_matrix()
    h3[12, 12] = 1.0

    h4 = zero_matrix()
    h4[13, 13] = 1.0

    return h_base, np.stack((h0, h1, h2, h3, h4), axis=0)


def apply_history_matrices(binary_input, global_input, include_history, h_base, h_builder):
    # Mirrors data_processing_pytorch.apply_history_matrices with symmetry=0 and no torch dependency.
    h_matrix = h_base + np.einsum("bi,ijk->bjk", include_history, h_builder)
    binary_input = np.einsum("bijk,bil->bljk", binary_input, h_matrix)
    padded_include = np.ones_like(global_input, dtype=np.float32)
    padded_include[:, : include_history.shape[1]] = include_history
    global_input = global_input * padded_include
    return binary_input, global_input


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--model-config-name", required=True)
    parser.add_argument("--pos-len", type=int, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--history-seed", type=int, required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--manifest-json", required=True)
    args = parser.parse_args()

    if args.model_config_name not in modelconfigs.config_of_name:
        raise ValueError(f"Unknown model config: {args.model_config_name}")
    model_config = modelconfigs.config_of_name[args.model_config_name]
    num_bin_features = modelconfigs.get_num_bin_input_features(model_config)
    num_global_features = modelconfigs.get_num_global_input_features(model_config)
    if num_bin_features != 22:
        raise ValueError("This sampler currently assumes 22 binary input features")
    if num_global_features != 19:
        raise ValueError("This sampler currently assumes 19 global input features")

    npz_files = sorted(
        glob.glob(os.path.join(args.input_dir, "**", "*.npz"), recursive=True)
    )
    if not npz_files:
        raise ValueError(f"No .npz files found under {args.input_dir}")

    packed_width = (args.pos_len * args.pos_len + 7) // 8
    row_counts = []
    field_info = {}
    for npz_file in npz_files:
        with np.load(npz_file) as npz:
            n = int(npz["globalInputNC"].shape[0])
            if npz["binaryInputNCHWPacked"].shape[2] != packed_width:
                raise ValueError(
                    f"{npz_file}: packed width {npz['binaryInputNCHWPacked'].shape[2]} "
                    f"does not match pos_len {args.pos_len}"
                )
            if not field_info:
                field_info = {
                    key: {
                        "shape": [int(x) for x in npz[key].shape[1:]],
                        "dtype": str(npz[key].dtype),
                    }
                    for key in npz.files
                }
            row_counts.append(n)

    total_rows = int(sum(row_counts))
    if total_rows < args.num_samples:
        raise ValueError(
            f"Only {total_rows} rows available, requested {args.num_samples}"
        )

    rng = np.random.default_rng(args.seed)
    selected_flat = np.sort(rng.choice(total_rows, size=args.num_samples, replace=False))

    offsets = np.cumsum([0] + row_counts)
    shard_groups = {}
    for flat_index in selected_flat:
        shard_idx = int(np.searchsorted(offsets[1:], flat_index, side="right"))
        row = int(flat_index - offsets[shard_idx])
        shard_groups.setdefault(shard_idx, []).append(row)

    h_base, h_builder = build_history_matrices()
    history_rng = np.random.default_rng(args.history_seed)
    should_stop_history = history_rng.random((args.num_samples, 5)) >= 0.98
    include_history = (
        np.cumsum(should_stop_history, axis=1, dtype=np.float32) <= 0.1
    ).astype(np.float32)

    n = args.num_samples
    binary_out = np.zeros(
        (n, num_bin_features, args.pos_len, args.pos_len), dtype=np.float32
    )
    global_out = np.zeros((n, num_global_features), dtype=np.float32)
    policy_out = np.zeros((n, 2, args.pos_len * args.pos_len + 1), dtype=np.float32)
    global_targets_out = np.zeros((n, GLOBAL_TARGETS_NC_CHANNELS), dtype=np.float32)
    score_out = np.zeros((n, 842), dtype=np.float32)
    value_out = np.zeros((n, 5, args.pos_len, args.pos_len), dtype=np.float32)
    qvalue_out = np.zeros((n, 3, args.pos_len * args.pos_len + 1), dtype=np.float32)
    include_history_out = np.zeros((n, 5), dtype=np.float32)

    output_pos = 0
    for shard_idx, rows in sorted(shard_groups.items()):
        rows_arr = np.array(rows, dtype=np.int64)
        with np.load(npz_files[shard_idx]) as npz:
            binary_packed = npz["binaryInputNCHWPacked"][rows_arr]
            binary = (
                np.unpackbits(binary_packed, axis=2)[:, :, : args.pos_len * args.pos_len]
                .reshape(rows_arr.size, num_bin_features, args.pos_len, args.pos_len)
                .astype(np.float32)
            )
            global_input = npz["globalInputNC"][rows_arr].astype(np.float32)
            include_group = include_history[output_pos : output_pos + rows_arr.size]
            binary, global_input = apply_history_matrices(
                binary, global_input, include_group, h_base, h_builder
            )
            end = output_pos + rows_arr.size
            binary_out[output_pos:end] = binary
            global_out[output_pos:end] = global_input
            policy_out[output_pos:end] = npz["policyTargetsNCMove"][rows_arr].astype(
                np.float32
            )
            global_targets_out[output_pos:end] = pad_global_targets_nc(
                npz["globalTargetsNC"][rows_arr]
            ).astype(np.float32)
            score_out[output_pos:end] = npz["scoreDistrN"][rows_arr].astype(np.float32)
            value_out[output_pos:end] = npz["valueTargetsNCHW"][rows_arr].astype(
                np.float32
            )
            qvalue_out[output_pos:end] = npz["qValueTargetsNCMove"][rows_arr].astype(
                np.float32
            )
            include_history_out[output_pos:end] = include_group
            output_pos = end

    os.makedirs(os.path.dirname(args.output_npz), exist_ok=True)
    np.savez(
        args.output_npz,
        binaryInputNCHW=binary_out,
        globalInputNC=global_out,
        includeHistory=include_history_out,
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
        "model_config_name": args.model_config_name,
        "pos_len": args.pos_len,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "history_seed": args.history_seed,
        "history_transform": "deterministic official history matrices, symmetry=0",
        "sampling": "uniform without replacement over global row index space",
        "source_archive": os.path.basename(args.source_archive),
        "source_archive_sha256": archive_hash,
        "num_shards": len(npz_files),
        "shard_list": shard_list,
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
        f"Rows={args.num_samples} shards={len(npz_files)} "
        f"total_rows_seen={total_rows} output_sha256={output_hash}"
    )


if __name__ == "__main__":
    main()
