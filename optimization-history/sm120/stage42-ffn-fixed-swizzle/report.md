# H42 result: fixed 18x37 FFN swizzle

## Decision

Rejected at the fixed-B13 S1 mechanism gate. No KataGo source or target
configuration changed.

## Evidence

- All 666 raw CTA coordinates match the generic panel-10 mapping exactly and
  form a complete 18x37 permutation.
- Boundary output is bit-identical (`0` mismatches, max absolute error `0`).
- Both variants use 136 registers/thread, 32,768 B dynamic shared memory, and
  zero stack/spills.
- Candidate SASS has 2,728 static instructions versus control 2,792 (`-64`).
- Interleaved S1 median regressed `36.708 -> 36.848 us` (`+0.383%`).

The instruction-count prediction held but did not reduce realized latency.
Per the frozen rule, NCU and natural whole-graph S2 were not run. Reopen only
if a source-counter profile identifies the swizzle prologue itself as a
material executed stall, rather than from static instruction count alone.

Last result timestamp: `2026-08-06 11:39:24 UTC`.
