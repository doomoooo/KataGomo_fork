# Stage 29: flat vec8 C768 affine-SiLU

## Decision

Rejected for the fixed B13/S2 target on top of the accepted trunk persisting-L2
baseline. The experimental runtime path was removed after the negative long
screen; the accepted half2 C384/C768 kernel remains unchanged.

The candidate flattened B13x361x768 into aligned half8 vectors, used one thread
per vector and launched 1,760 CTAs of 256 threads instead of one 384-thread CTA
per row (4,693 CTAs). It retained four half2 FMAs and the same scalar FP32 SiLU
calculation per vector.

## Results

Short 400/25 ABBA:

- control: 3835.157, 3832.733 nn/s; mean 3833.945;
- candidate: 3834.546, 3840.432 nn/s; mean 3837.489;
- mean delta: +0.092%, with mixed adjacent-pair direction.

Long 1000/30 symmetric ABBA/BAAB:

- control: 3812.478, 3800.947, 3799.419, 3787.746; mean 3800.148;
- candidate: 3812.357, 3801.490, 3780.476, 3779.395; mean 3793.430;
- mean delta: -0.177%; only one of four adjacent comparisons was positive.

The predeclared whole-network performance gate failed, so Nsys, NCU and full
accuracy replay were not run. Reopen only if the vector schedule is fused with
an adjacent outer projection or a new mechanism changes the S2 resource phase;
another standalone flattening/block-size sweep is not justified.
