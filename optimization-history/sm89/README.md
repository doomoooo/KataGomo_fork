# SM89 optimization evidence

This tree contains the RTX 4090 / compute capability 8.9 optimization record.

Start with:

1. `HISTORY.md` for the chronological StageXX decisions.
2. `ITERATION-PROTOCOL.md` for measurement and acceptance rules.
3. `INTRINSIC-S2-AUDIT.md` for the intrinsic/fusion review.
4. `stage*/final-decision-summary.json` for machine-readable stage outcomes.

Stage directories run from Stage0 through Stage70 plus Stage72. Stage71 was a
GTP graph/eager and thermal-attribution investigation rather than a standalone
kernel directory; its evidence is under `gtp-pipeline-gap-investigation/`, in
particular `default-boost-peak-stage71/`.

`explicit-compute-stream/`, `infer-event-pipeline-stage68/`, and
`gtp-pipeline-gap-investigation/` cover frontend/stream integration after the
kernel optimization stages. `tensorrt-baseline/` is comparison evidence, not
part of the CUDA plan implementation.
