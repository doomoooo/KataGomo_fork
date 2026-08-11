# Stage 54: out-projection / following FFN RMSNorm boundary

## Decision

Rejected. No repository source was changed and no whole-graph profile was
triggered. The accepted/default baseline remains commit `3526b13`, whose
performance baseline is the Stage-47 accepted graph at commit `acf588c`.

## cuDNN Graph probe

cuDNN 9.25 rejected both the direct rank-3 description and the canonical
rank-4 description with a zero-copy view reshape. Debug logging showed the
fully formed four-operation graph

`matmul -> view-only reshape -> pointwise add -> RMSNorm`

and failed operation-graph finalization with
`CUDNN_STATUS_NOT_SUPPORTED_GRAPH_PATTERN` / `matcher->matchPattern`. There is
therefore no executable plan to profile with Nsys/NCU.

## CuTe folded-RMS probe

The fallback did not guess a full-C384 output tile. Because Transformer FFN
RMSNorm has beta=0, gamma was folded into the paired linear/gate weights. A
scale-only kernel produced one FP32 inv_rms value per row, and the otherwise
unchanged accepted Stage-47 CuTe FFN multiplied its two accumulator fragments
by that row value before SwiGLU.

Correctness on deterministic random FP16 data was finite with
`max_abs=0.0078125` and `RMSE=0.000181115` versus the accepted rounded RMSNorm
plus CuTe FFN boundary. This was sufficient for a performance probe but was
not claimed byte-identical.

### Nsys / NCU mechanism

- Nsys over mixed S1/S2 launches measured the scale-only kernel at 2.905 us
  average versus 3.437 us for full RMSNorm.
- Targeted NCU confirmed the scale kernel is strictly lighter: 2.432 versus
  3.296 us and 16 versus 36 registers/thread, with identical 1174-CTA grid,
  128-thread blocks, zero dynamic shared memory, and 0.58 waves/SM.
- The CuTe FFN itself retained exactly the same launch resources in both
  versions: 96 registers/thread, 50.176 KiB dynamic shared memory, 340 CTAs,
  288 threads, one wave/SM. The extra scale load/multiply made the targeted
  NCU launch 30.176 us versus 27.360 us. It was therefore not a strictly
  resource-improved complete boundary.

### Final boundary gate

One no-profiler 400-iteration ABBA run reported:

| mode | control A/B (us) | candidate B/B (us) | pooled delta |
|---|---:|---:|---:|
| S1 | 31.2626 / 31.2726 | 31.3257 / 31.3910 | +0.2903% slower |
| S2 wall | 61.3118 / 61.3045 | 61.3265 / 61.3166 | +0.0219% slower |

The candidate fails both the default S2 gate and the S1/resource retention
gate. It is not integrated or committed.

## Artifacts

- `cudnn_outproj_rmsnorm_probe.cu`
- `generate_folded_rms_ffn_probe.py`
- `folded_rms_ffn_probe.cu`
- `nsys-folded-rms-boundary.nsys-rep`
- `ncu-control-ffn.ncu-rep`, `ncu-folded-rms-ffn.ncu-rep`
- `ncu-control-rmsnorm.ncu-rep`, `ncu-candidate-rmsscale.ncu-rep`
- `folded-rms-boundary-abba.txt`

Reopen only for an implementation that can normalize before MMA without
duplicating a full row reduction per output tile, or for a proven SM120
cluster/DSM reduction design whose resource signature is established before
whole-graph integration.
