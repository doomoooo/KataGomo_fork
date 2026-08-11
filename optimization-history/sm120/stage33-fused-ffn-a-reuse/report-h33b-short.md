# H33b: resource-positive retention and short whole-graph screen

## Status

**Advanced to the long whole-graph gate.** H33's original homogeneous-S2
rejection remains immutable. Under the revised retention policy, H33b retained
the bit-exact, resource-positive A-reuse candidate long enough to test the
heterogeneous graph. The short B13/19x19/S2 screen is positive and all four
adjacent comparisons agree.

This report intentionally stops before the long run. No 1000-iteration result,
full replay, Nsys, or full-forward NCU was collected in this step.

## Mixed-peer diagnostics

The diagnostics used the standalone control/candidate sources from H33, the
exact current linear2 AOT source at SHA256
`811b8c3a3288305dc80e61398b1ddbbbfa402c8b21869d1d10ff1138653a2f42`,
and the current ordinary out-projection boundary as direct FP16
`cublasHgemm(M=384,N=4693,K=384,beta=1)`. Measurements used eight symmetric
alternating ABBA cycles, 40 warmups, and 600 iterations.

| Mixed peer | metric | control us | candidate us | delta |
|---|---|---:|---:|---:|
| current linear2 AOT | fused FFN stream | 48.016 | 47.300 | -1.492% |
| current linear2 AOT | peer stream | 45.407 | 45.065 | -0.752% |
| current linear2 AOT | pair makespan | 48.274 | 47.806 | -0.969% |
| ordinary out-projection | fused FFN stream | 40.664 | 38.371 | -5.638% |
| ordinary out-projection | peer stream | 28.862 | 36.041 | +24.871% |
| ordinary out-projection | pair makespan | 40.682 | 38.757 | -4.730% |

The linear2 coexistence signal is positive. The out-projection result shows a
large redistribution: fused FFN remains the pair critical path and becomes
faster, while the cuBLAS peer becomes slower. This is diagnostic evidence that
the graph-level result, rather than either peer duration alone, must adjudicate
the candidate.

## Integration

The already-tested source was integrated without a new tile or schedule:

- Added `fused_ffn_b13_a_reuse.cu` to the SM120 AOT sources.
- Added `cudaUseFusedFFNAReuseSm120`, default `false`.
- The existing S1 schedule remains higher priority.
- The new path is reachable only inside the existing exact B13, 19x19, FP16
  fused-FFN dispatch; transformer NHWC is enforced by the enclosing backend.
- With the switch disabled, the existing `launchFusedFFNB13` remains the
  fallback.
- The target configuration was not changed.

Build completed successfully. Binary SHA256 is
`c60db86740a49516b6aae1067a2bd72df62e8a1b1da7a5f20ccc7ccb559e3091`;
`git diff --check` is clean. Candidate smoke logs confirm the
`A-fragment reuse` path was active on both server threads.

## Short S2 whole graph

Frozen command boundary: current target config, B13, 19x19, S2, CUDA-event
forward timing. After one 500/30 control thermal prime, the timed order was
`A-B-B-A-B-A-A-B`, where A disables and B enables only
`cudaUseFusedFFNAReuseSm120`. Each timed arm used 400 iterations and 25 warmups.

| Variant | throughput samples (nnEval/s) | mean (nnEval/s) |
|---|---|---:|
| control | 3842.775, 3839.012, 3858.382, 3843.412 | 3845.895 |
| candidate | 3867.054, 3888.532, 3895.355, 3872.801 | 3880.935 |

- Mean improvement: `+0.911%`.
- Adjacent candidate deltas: `+0.632%`, `+1.290%`, `+0.958%`, `+0.765%`.
- Positive adjacent comparisons: `4/4`.

This passes H33b's short gate by both conditions: positive mean and at least
three positive adjacent comparisons. The next authorized action is the
symmetric 1000/30 ABBA/BAAB run. Its result, not this short screen, decides
performance retention.

## Artifacts

- `hypothesis-h33b.md`: preregistered follow-up
- `raw-mixed-peers.json`: complete mixed-peer samples, SHA256
  `8d212441804091f830ceabe6338565a005cc137deca68af9349205d0444ccce4`
- `mixed_peers.py`, `linear2_peer.generated.cu`, `cublas_peer.cu`: exact-peer harness
- `integration-manifest.json`: switch, dispatch, build, and hashes
- `run-h33b-short-abba.sh`: complete short sequence
- `short-h33b/`: raw logs and results
- `short-h33b/summary.json`: gate summary, SHA256
  `7c0833321c4bb977b04f960f34e668346f30cbf24369d97949a924ba41a0a13c`

