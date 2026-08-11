# Stage 28: persisting-L2 C768 trunk window

## Decision

Accepted for the fixed RTX 5090 D B13/S2 target. The code path defaults off and
the target config explicitly enables `cudaUsePersistingL2Trunk=true`.

The exact per-stream window is `13 * 361 * 768 * 2 = 7,208,448` bytes. Two
streams request 14,416,896 bytes of device-global persisting-L2 set-aside; the
runtime granted 18,874,368 bytes, so both stream windows use hit ratio 1.0.
The access-policy window starts at the real `trunkScratch` lifetime and is
cleared after trunk-tip normalization. Inner C384 windows remain disabled.

## Performance

Short B13/S2 ABBA:

- control: 3824.650, 3805.781 nn/s; mean 3815.216;
- candidate: 3854.218, 3856.647 nn/s; mean 3855.432;
- mean delta: +1.054%.

Long 1000/30 symmetric ABBA/BAAB after thermal priming:

- control: 3721.887, 3716.001, 3700.924, 3700.444; mean 3709.814;
- candidate: 3741.636, 3746.294, 3740.232, 3728.386; mean 3739.137;
- mean delta: +0.790%; every adjacent control/candidate comparison is positive.

Nsys kept the kernel count identical at 52,656. Total GPU kernel duration was
674.161 ms for control and 666.738 ms for candidate, a 1.10% reduction. The
only added runtime work is the expected stream access-policy attribute calls.

## NCU mechanism

Two exact B13/grid=4693 C768 `affineSiluHalf2Kernel<768>` launches per arm:

| metric | control | candidate | delta |
|---|---:|---:|---:|
| kernel time | 6.592 us | 5.888 us | -10.68% |
| L2 sector hit rate | 52.53% | 94.60% | +42.07 pp |
| DRAM bytes | 7.214 MB | 5.073 MB | -29.68% |
| evict-last sectors | 0 | 225,264 | active |

This supports the declared cache-residency mechanism rather than a launch-count
or arithmetic-path artifact.

## Accuracy and artifacts

The full 8,192-row replay is byte-identical to the accepted Stage27 RMSNorm
vec8 output. Both files have SHA256
`ed0ed80848d752bc6d64995e91f9bada55c059b5e55ac5bcccb13bf28a3e1a02`.

- Hypothesis: `hypothesis-persisting-l2-trunk.md`
- Short benchmark: `persisting-l2-trunk-short-abba/`
- Long benchmark: `persisting-l2-trunk-long/`
- Nsys: `persisting-l2-trunk-nsys/`
- NCU: `persisting-l2-trunk-ncu/*-b13.{ncu-rep,csv}`
- Accuracy: `persisting-l2-trunk-accuracy/`
