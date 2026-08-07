#!/usr/bin/env python3
"""Compare two replaynn .krnn dumps and produce the SKILL.md accuracy metrics.

The .krnn format is produced by `katago replaynn`: KRNN magic + JSON metadata +
float32 sections (policy logits, policy pass logits, value logits, score-value
logits, ownership, targets, and inputs).

Usage:
  python compare_replay_krnn.py \
    --reference replay-fp32-full19.krnn \
    --candidate replay-fp16-full19.krnn \
    --output /workspace/results/accuracy/replay-fp16-vs-fp32.json
"""

import argparse
import json
import os
import struct
from datetime import datetime, timezone

import numpy as np


def read_krnn(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"KRNN":
            raise ValueError(f"{path}: bad magic {magic!r}")
        (meta_len,) = struct.unpack("<I", f.read(4))
        meta = json.loads(f.read(meta_len))
        blob_start = f.tell()

        num_rows = int(meta["numRows"])
        pos_len = int(meta["posLen"])
        total_bytes = sum(int(s["bytes"]) for s in meta["sections"])
        f.seek(blob_start)
        blob = np.fromfile(f, dtype=np.float32, count=total_bytes // 4)
        if blob.size * 4 != total_bytes:
            raise ValueError(f"{path}: blob size mismatch")

        sections = []
        offset = 0
        for sec in meta["sections"]:
            dim = int(sec["dim"])
            n = num_rows * dim
            sections.append(blob[offset : offset + n].reshape(num_rows, dim))
            offset += n
    return meta, sections


def stable_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return x - np.log(np.sum(e, axis=axis, keepdims=True))


def kl_div(p, q):
    return np.sum(p * (np.log(p + 1e-30) - np.log(q + 1e-30)), axis=-1)


def jsd(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_div(p, m) + kl_div(q, m))


def normalize_target(target):
    sums = target.sum(axis=-1, keepdims=True)
    return np.where(sums > 0, target / np.maximum(sums, 1.0), 1.0 / target.shape[-1])


def weighted_mean(values, weights):
    wsum = weights.sum()
    if wsum <= 0:
        return float("nan")
    return float(np.sum(values * weights) / wsum)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def top1_agreement(a, b):
    return float(np.mean(np.argmax(a, axis=-1) == np.argmax(b, axis=-1)))


def policy_metrics(sections, pos_area):
    policy = sections[0].reshape(sections[0].shape[0], pos_area, 2)
    pass_logits = sections[1]
    pred0 = np.concatenate([policy[:, :, 0], pass_logits[:, 0:1]], axis=1)
    pred1 = np.concatenate([policy[:, :, 1], pass_logits[:, 1:2]], axis=1)
    target_policy = sections[5].reshape(sections[5].shape[0], 2, pos_area + 1)
    target0 = normalize_target(target_policy[:, 0, :])
    target1 = normalize_target(target_policy[:, 1, :])
    return pred0, pred1, target0, target1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ref_meta, ref = read_krnn(args.reference)
    cand_meta, cand = read_krnn(args.candidate)
    if ref_meta["numRows"] != cand_meta["numRows"]:
        raise ValueError("reference/candidate row counts differ")
    if ref_meta["posLen"] != cand_meta["posLen"]:
        raise ValueError("reference/candidate posLen differ")

    n = int(ref_meta["numRows"])
    pos_len = int(ref_meta["posLen"])
    pos_area = pos_len * pos_len
    policy_len = pos_area + 1

    ref_pred0, ref_pred1, ref_target0, ref_target1 = policy_metrics(ref, pos_area)
    cand_pred0, cand_pred1, cand_target0, cand_target1 = policy_metrics(cand, pos_area)

    ref_probs0 = stable_softmax(ref_pred0)
    cand_probs0 = stable_softmax(cand_pred0)
    ref_probs1 = stable_softmax(ref_pred1)
    cand_probs1 = stable_softmax(cand_pred1)

    ref_value_probs = stable_softmax(ref[2])
    cand_value_probs = stable_softmax(cand[2])

    target_global_ref = ref[6]
    target_global_cand = cand[6]
    global_weight = target_global_ref[:, 25]

    ce0_ref = -np.sum(
        ref_target0 * log_softmax(ref_pred0, axis=1), axis=1
    )
    ce0_cand = -np.sum(
        cand_target0 * log_softmax(cand_pred0, axis=1), axis=1
    )

    result = {
        "createdUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reference": args.reference,
        "candidate": args.candidate,
        "referenceRevision": ref_meta.get("revision"),
        "candidateRevision": cand_meta.get("revision"),
        "corpus": ref_meta.get("corpus"),
        "numRows": n,
        "posLen": pos_len,
        "policy": {
            "p0lossReferenceWeighted": weighted_mean(ce0_ref, global_weight),
            "p0lossCandidateWeighted": weighted_mean(ce0_cand, global_weight),
            "p0lossReferenceUnweighted": float(np.mean(ce0_ref)),
            "p0lossCandidateUnweighted": float(np.mean(ce0_cand)),
            "top1VsReference": top1_agreement(cand_pred0, ref_pred0),
            "optimisticTop1VsReference": top1_agreement(cand_pred1, ref_pred1),
            "top1VsTarget": top1_agreement(cand_pred0, ref_target0),
            "optimisticTop1VsTarget": top1_agreement(cand_pred1, ref_target1),
            "probabilityRmse": rmse(cand_probs0, ref_probs0),
            "optimisticProbabilityRmse": rmse(cand_probs1, ref_probs1),
            "totalVariation": float(
                np.mean(0.5 * np.sum(np.abs(cand_probs0 - ref_probs0), axis=1))
            ),
            "jsd": float(np.mean(jsd(cand_probs0, ref_probs0))),
        },
        "value": {
            "outcomeRmse": rmse(cand_value_probs, ref_value_probs),
            "rawLogitsRmse": rmse(cand[2], ref[2]),
        },
        "score": {
            "meanRmse": rmse(cand[3][:, 0], ref[3][:, 0]),
            "all6Rmse": rmse(cand[3], ref[3]),
        },
        "ownership": {
            "rawLogitsRmse": rmse(cand[4], ref[4]),
            "sigmoidRmse": rmse(
                1.0 / (1.0 + np.exp(-cand[4])),
                1.0 / (1.0 + np.exp(-ref[4])),
            ),
        },
        "maxPolicyAbsError": {
            "row": int(np.unravel_index(
                np.abs(cand_probs0 - ref_probs0).argmax(), cand_probs0.shape
            )[0]),
            "move": int(np.unravel_index(
                np.abs(cand_probs0 - ref_probs0).argmax(), cand_probs0.shape
            )[1]),
            "value": float(np.abs(cand_probs0 - ref_probs0).max()),
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
