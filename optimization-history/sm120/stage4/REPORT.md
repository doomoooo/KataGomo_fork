# Stage 4: fixed-S361 FA4 tail mask

Status: rejected (2026-08-05 UTC).

The experiment specialized flash-attn's non-causal sequence tail mask for the
only AOT shape in scope: 19x19, S361, tile N=128. The final K tile has exactly
105 valid columns, so the candidate replaced dynamic `seqlen_k - n_block*128`
limit calculation with the constant 105. C++ dispatch still gated the AOT path
on `seqLen == 361`; all other sizes retained the official fallback.

The candidate object SHA256 was
`8b865c696c40d785eb1db050659ed70eefe8c7a0f7337d17f9f51f4b8359f190`.
Its standalone attention output was byte-identical to the accepted generic
both16 object and passed `attention_ref` with max absolute error 2.44e-4.

Same-lock, 20,000-iteration-per-repeat micro ABBA/BAAB result:

| mode | median-of-run medians (us) | mean (us) | range (us) |
|---|---:|---:|---:|
| accepted generic mask | 11.80745 | 11.80715 | 11.80287-11.81085 |
| fixed S361 mask | 11.83552 | 11.83507 | 11.83122-11.83804 |

The fixed mask is 0.236% slower. It failed the predeclared isolated-performance
gate, so no whole-network benchmark or full 8,192-row replay was run. The
candidate is rejected and the checked-in/default AOT object remains the
generic-mask both16 Stage-3 object (SHA256
`9b0c9b33c7bee1e61350cb0c3f422a58a5aaf2b20c2b426e638c0253f84398ac`).

Reopen only if a future FA4 version exposes a truly static S dimension that
also eliminates dynamic loop/block scheduling, rather than replacing only the
small tail-limit expression.
