# Stage 29: persisting-L2 C384 inner window

## Evidence verdict

Accepted for the fixed RTX 5090 D, exact 19x19, B13, FP16 NHWC, S2 target.
The code default remains false and the target config explicitly enables
`cudaUsePersistingL2Inner=true`.

The accepted Stage28 trunk window is the common baseline in every comparison.
Control/candidate differ only in `cudaUsePersistingL2Inner=false/true`.

## Implementation and quota

The implementation reproduces the RTX 4090 Stage21 lifetime boundary in the
official block graph:

1. `normActConv1.norm` runs while the C768 trunk window is active.
2. Immediately before `normActConv1.conv` writes `mid`, the stream switches to
   the exact C384 `mid` window.
3. The C384 window remains active across all six inner transformer blocks and
   `normActConv2.norm`.
4. After enqueueing that norm and before `normActConv2.conv` accumulates into
   the outer residual, the stream restores the C768 `trunkBuf` window.

Window and device-global quota sizes are:

```text
C768 trunk per stream = 7,208,448 bytes
C384 inner per stream = 3,604,224 bytes
exact S2 request       = 21,625,344 bytes
runtime grant          = 25,165,824 bytes
trunk/inner hit ratio  = 1.0 / 1.0
```

The control still requests/grants `14,416,896 / 18,874,368` bytes. The
candidate adds exactly 22 stream-attribute calls per forward per stream, one
switch and one restore for each of 11 nested blocks. No arithmetic kernel,
kernel geometry, allocation, copy, or synchronization changes. In particular,
`cudabackend_sm120_kernels.cu` and affine/RMSNorm vec8 code were not modified.

Build succeeded with binary SHA256
`d79eb507fcc7ea26644c92d7df715cbba5c1c90826d51a3bac69e3763b260e7f`;
`git diff --check` is clean.

## Throughput

Short B13/S2 ABBA after control thermal priming:

- control: `3782.747`, `3801.899`; mean `3792.323 nn/s`;
- candidate: `3858.451`, `3854.026`; mean `3856.238 nn/s`;
- mean delta: `+1.685%`.

Clean long 1000/30 ABBA/BAAB round 1:

- control mean: `3796.283 nn/s`;
- candidate mean: `3852.010 nn/s`;
- delta: `+1.468%`;
- ABBA direction: `+1.296%`; reverse BAAB direction: `+1.640%`;
- all four adjacent control/candidate pairs are positive.

Independently primed long round 2 with GPU state captured after each arm:

- control mean: `3817.100 nn/s`;
- candidate mean: `3871.555 nn/s`;
- delta: `+1.427%`;
- ABBA direction: `+1.512%`; reverse BAAB direction: `+1.342%`;
- all four adjacent pairs are positive:
  `+1.691%`, `+1.332%`, `+1.256%`, `+1.427%`.

Pooling the two clean rounds equally gives `3806.692 -> 3861.783 nn/s`,
`+1.447%`.

The separate `long/` run is retained but excluded: its first three timed arms
overlap the NCU guard artifact interval (`07:31:08` through `07:32:03` UTC),
and the first two results are anomalously low (`3543.656`, `3570.511 nn/s`)
before later arms recover to the normal `3809-3859 nn/s` range. It is a
contaminated run, not contrary performance evidence.

## Nsys

The 30-forward-per-stream traces are phase matched: median stream offset is
`3.665 us` for control and `3.760 us` for candidate.

| metric | control | candidate | result |
|---|---:|---:|---:|
| benchmark throughput | 3875.308 nn/s | 3939.513 nn/s | +1.657% |
| kernel launches | 52,656 | 52,656 | unchanged |
| memcpy / memset | 2,742 / 276 | 2,742 / 276 | unchanged |
| stream attribute calls | 268 | 3,216 | exact 2 -> 24 per forward/stream |
| summed kernel duration | 668.618 ms | 663.144 ms | -0.819% |
| B13 C384 affine median | 4.128 us | 4.064 us | -1.550% |
| B13 C768 affine median | 7.552 us | 7.488 us | -0.847% |

Both traces retain 344 kernels per forward on both streams. Synchronization
counts are unchanged (`cuCtxSynchronize=1`, `cudaDeviceSynchronize=32`).

## NCU mechanism

Exact B13/grid=4693 C384 consumer, two launches per arm:

| metric | control | candidate | delta |
|---|---:|---:|---:|
| kernel time | 4.784 us | 4.448 us | -7.02% |
| L2 sector hit rate | 53.29% | 96.97% | +43.67 pp |
| DRAM bytes | 5.106 MB | 2.396 MB | -53.07% |
| evict-last sectors | 0 | 112,632 | policy active |

The initial two-sample C768 guard had noisy time/DRAM direction, so it was not
used alone. The eight-sample exact B13/grid=4693 guard resolves it:

| metric | control | candidate | delta |
|---|---:|---:|---:|
| kernel time | 6.576 us | 6.096 us | -7.30% |
| L2 sector hit rate | 80.98% | 82.31% | +1.33 pp |
| DRAM bytes | 3.068 MB | 3.038 MB | -0.98% |
| evict-last sectors | 225,264 | 225,264 | trunk policy retained |

Thus the C384 cache-residency mechanism is present and the accepted C768 trunk
window is not damaged by switching.

## Accuracy

An independently completed 8,192-row replay produced `445,383,334` bytes and
is byte-identical to the accepted Stage28 trunk-L2 replay. Both SHA256 values
are:

```text
ed0ed80848d752bc6d64995e91f9bada55c059b5e55ac5bcccb13bf28a3e1a02
```

The target config and optimization history record this candidate as accepted.

## Artifacts

- Hypothesis: `hypothesis-persisting-l2-inner.md`
- Short screen: `short-abba/`
- Nsys and interference analysis: `nsys/`
- NCU C384 and C768 reports: `ncu/`
- Clean long rounds: `long-abba-baab/`, `long-r2/`
- Contaminated retained run: `long/`
- Byte-identical replay: `accuracy/`
