# SM120 optimization evidence

This tree contains the RTX 5080/5090D / compute capability 12.0 optimization
record.

Start with:

1. `HISTORY.md` for the chronological accepted/rejected decisions.
2. `5080-CROSSCHECK.md` for the cross-device review.
3. `stage*/REPORT.md` or `stage*/report.md` for each experiment's conclusion.
4. `cross-batch-search/` for the historical B4-B32 scanner and plan evidence.

The later descriptive stage names are intentional: several Stage29–57
investigations had parallel sub-experiments. Their directory names preserve the
subject while `HISTORY.md` provides the total ordering.

Profiler traces, generated AOT objects, historical binaries, and failed search
artifacts are retained as evidence. They are not runtime dependencies of the
current production plan.
