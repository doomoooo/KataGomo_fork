# Stage 44 completion: RTX5090D fixed-B13 packed QKV

## Result

Accepted for the exact deployment target: RTX 5090 D, 19x19, B13, natural
S2, FP16 with FA4 `both16`.

The accepted candidate is
`qkv-m128-n128-k64-s2-cute-atom4x2-packed`. It writes row-packed Q/K/V,
uses the packed batch-shared half2 RoPE path, and is consumed directly by FA4
through dynamic strides. The generated artifact is marked as the automatic
winner only for RTX5090D/B13/S2. A build with the stub artifact or another
GPU/batch/topology continues to select the previously accepted planar tactic.

Implementation preparation was committed at `01f140c`. The common-wall
measurement correction was committed separately at `36627fb`. Automatic
deployment promotion was committed at `ee4d1d8`. The profiled
binary SHA256 is
`17b1b9a3c6a13ca73b018c95a9967b0db9ad2072f5845f5fc1f2333993edd97e`.
After adding a generator-side guard that forbids marking non-B13 artifacts as
automatic winners, the final rebuilt binary SHA256 is
`b1cddcf8adc9dc09be280a975e8477faaf2f5643ad5df0b70c6cc85884d827fa`;
the linked AOT object and bridge hashes are unchanged, and a final 50-iteration
auto-selection smoke measured `3962.015` common-wall physical nnEval/s.

## Complete-boundary evidence

The valid S1 Nsys comparison used the same binary and measured 20 complete
forwards with 344 kernels/forward. The first attempted trace was discarded
because an override changed automatic tactic selection and produced 410
kernels/forward. It has no decision authority.

| natural component | planar mean | packed mean | change |
|---|---:|---:|---:|
| QKV | 19.228 us/block | 14.164 us/block | -26.336% |
| RoPE | 3.845 us/block | 5.624 us/block | +46.279% |
| FA4 | 11.866 us/block | 11.902 us/block | +0.303% |
| QKV start -> FA4 end | 35.451 us/block | 32.214 us/block | -9.131% |
| sum of 33 natural boundaries/forward | 1169.877 us | 1063.050 us | -106.827 us |
| complete forward GPU span | 4166.619 us | 4071.045 us | -2.294% |

This comparison directly measures the composed boundary. No independently
replayed NCU durations were added together.

Targeted NCU used two samples per kernel. Packed QKV reduced registers from
136 to 107 and had zero local spilling in both modes. Active warps increased
from 8.31% to 18.47%, eligible warps/scheduler from 0.0695 to 0.2082, and SM
throughput from 43.07% to 57.98%. The tradeoff is explicit: dynamic shared
memory increased from 65.536 KiB to 99.328 KiB and waves/SM fell from 1.96 to
1.0. The candidate is therefore mechanism-positive, not strictly dominant in
every resource.

## Natural S2 deployment result

The benchmark harness was first corrected to count completed physical B13
launches over one common timed wall. It no longer uses the sum of per-lane
median throughputs as the deployment metric. Eight 100-iteration same-binary
runs were executed as ABBA followed by reversed BAAB; stream priority was
unchanged and `CUDA_DEVICE_MAX_CONNECTIONS` was unset.

| mode | common-wall physical nnEval/s mean | range |
|---|---:|---:|
| Stage-38 planar control | 3896.899 | 3888.635–3901.769 |
| packed candidate | 3920.182 | 3912.162–3936.928 |

Mean improvement is **+0.5975%**. All four adjacent comparisons are positive:
`+0.416%`, `+0.467%`, `+0.605%`, and `+0.901%`; their median is `+0.536%`.
The obsolete median-sum metric differs by about 0.05% here and does not alter
the result.

The short auto-selection smoke confirms that the no-override Stage-38 config
selects the packed QKV and packed RoPE paths. An explicit planar tactic still
selects the old path.

## Accuracy

The complete 8,192-row fixed 19x19 corpus passed every established all-head
gate against the full-FP32 reference:

- policy top-1: `99.7803%` (gate `>=99.70%`)
- optimistic policy top-1: `99.7070%` (gate `>=99.60%`)
- policy probability RMSE: `0.000101885`
- outcome RMSE: `0.00227039`
- score mean RMSE: `0.00187320`
- ownership sigmoid RMSE: `0.000245847`
- maximum policy absolute error: `0.0184333`

## Accepted full-graph checkpoint and next target

The post-acceptance S2 Nsys trace has 344 kernels/forward on both streams. The
matching accepted S1 NCU snapshot contains exactly 344 ordinal rows and uses a
wide, shallow four-section capture; detailed NCU was limited to QKV and RoPE.

The new S2-vs-S1 interference ranking is led by:

1. attention out-projection residual: `12.905 ms` accumulated excess;
2. fused FFN: `12.178 ms`;
3. linear2 residual: `11.906 ms`;
4. packed QKV: `11.331 ms`;
5. FA4: `7.435 ms`.

The accepted S1 NCU checkpoint shows linear2 at 162 registers, 65.536 KiB
dynamic shared memory, 0.65 waves/SM, 8.27% active warps, and only 0.0918
eligible warps/scheduler. Attention out-projection uses 164 registers, 81.920
KiB dynamic shared memory, and 0.87 waves/SM. These confirm that the next work
should be a boundary/fusion mechanism, not another ordinary standalone GEMM
tactic.

Kimi K3's independent read-only audit ranked postConv-to-next-C768-BN/SiLU
fusion first and final-inner-linear2-to-outer-C384-BN/SiLU fusion second. Its
pre-GPU negative prior for packed QKV was falsified by this stage, but its next
two rankings agree with the new profiler evidence. The next stage should start
from those two composed boundaries, with the 4090 measured transfer evidence
as prior and natural 5090D S2 as the deployment decision.

## Primary artifacts

- `hypothesis.md`: frozen target, mechanism, and gates
- `nsys/attention-boundary-comparison.json`: valid natural S1 boundary result
- `ncu/{control,candidate}-{qkv,rope}.ncu-rep`: targeted mechanism captures
- `s2-abba/summary.json`: common-wall natural S2 result
- `accuracy/packed-qkv-vs-fp32.json`: full accuracy report
- `accepted-profile/nsys/accepted-s2-analysis.{json,md}`: fresh S2 graph
- `accepted-profile/ncu/accepted-s1-full-forward.ncu-rep`: fresh 344-ordinal NCU
- `accepted-profile/ncu/accepted-s1-full-forward-summary.json`: resource summary
