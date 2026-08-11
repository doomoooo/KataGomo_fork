# H39b result: remove the post-commit FFN barrier

## Decision

Rejected at the fixed-B13 S1 mechanism gate. No KataGo source or target
configuration changed.

## Evidence

- Boundary output: bit-identical over 5,406,336 FP16 elements (`0` mismatches,
  max absolute error `0`).
- ptxas: both variants use 136 registers/thread, one barrier resource, zero
  stack and zero spills.
- SASS: control has five static `BAR.SYNC.DEFER_BLOCKING` sites; candidate has
  four, proving the intended main-loop instruction was removed.
- Interleaved S1 median: `36.945 -> 37.681 us`, a `1.993%` regression.

The removed barrier was logically unnecessary for correctness, but was useful
to the realized asynchronous-copy schedule. Since the candidate is coherently
slower in S1 despite strictly fewer executed barrier instructions, NCU and real
whole-graph S2 were not run. Reopen only as part of a different copy-group
schedule whose wait/commit distances are changed together.

Artifacts: `build-manifest.json`, `build.stderr`, `raw-s1.json`, generated
translation units, and the exact shared library. Last result timestamp:
`2026-08-06 11:19:20 UTC`.
