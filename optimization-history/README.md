# CUDA optimization evidence archive

This directory is the Git mirror of the SM89 and SM120 CUDA optimization
archive. It contains every Markdown report/history plus the valid JSON files
that represent summaries, decisions, comparisons, correctness results, or
profiler analyses from the canonical local `/workspace/results` tree. It
deliberately contains no other file type. Raw profiler databases, CSV traces,
logs, individual benchmark captures, search spaces, plans, manifests,
generated code, and historical binaries remain outside Git.

## Layout

- `sm89/`: RTX 4090 / SM89 optimization history and StageXX evidence.
  Start with `sm89/HISTORY.md`, `sm89/ITERATION-PROTOCOL.md`, and each stage's
  `final-decision-summary.json`.
- `sm120/`: RTX 5080/5090D / SM120 optimization history and StageXX evidence.
  Start with `sm120/HISTORY.md` and the `REPORT.md` or `report.md` in each
  accepted/rejected stage.
- `shared/accuracy/`: cross-backend numerical comparison reports.
- `shared/baselines/`: official CUDA and TensorRT baseline scans.
- `research/luminal/`: Markdown/JSON metadata from separate compiler/backend
  feasibility experiments; these are not accepted CUDA backend tactics.
- `docs/`: cross-architecture histories, portability notes, scheduler design,
  and the frozen scanner handover.
- `FILE-CATEGORY-SIZES.md`: size inventory and Git/archive policy.

## Evidence policy

Historical command lines and manifests are immutable provenance. They may
contain the absolute path that was valid when the run was performed; importing
the archive does not rewrite those records. New source references use this
repository's `optimization-history/sm89` and `optimization-history/sm120`
layout.

The corresponding raw profile or binary must not be deleted merely because it
is absent from Git. Git suitability and archival value are separate decisions.
