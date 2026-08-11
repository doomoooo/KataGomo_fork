# H40 result: fast FP32 quotient in fused FFN

## Decision

Rejected at the compile/resource mechanism gate. No KataGo source or target
configuration changed, and no GPU performance or full replay was run.

## Evidence

| implementation | registers/thread | stack/spill |
|---|---:|---:|
| accepted precise quotient control | 136 | 0 B |
| CUDA `__fdividef` | 168 | 64 B store + 64 B load |
| inline PTX `div.approx.ftz.f32` | 145 | 0 B |
| inline PTX with `__maxnreg__(136)` | 136 | 36 B store + 36 B load |

The fast quotient removes precise-divide refinement, but ptxas keeps more of
the fully unrolled epilogue live. No implementation simultaneously preserved
the accepted 136-register, zero-spill resource signature. This directly
violates the frozen mechanism gate, so boundary performance, NCU, whole-graph
S2, and accuracy replay were not used.

Reopen only with a separately ablated epilogue live-range/unroll change that
returns to at most 136 registers and zero spills before timing. Last artifact:
`2026-08-06 11:35:01 UTC`.
