# H42: Fixed 18x37 FFN CTA swizzle

## Frozen protocol

- RTX 5090 D, fixed B13/19x19 accepted A-reuse FFN kernel.
- Replace only generic `rasterization2DRow<10>()` with an exactly equivalent
  18x37, panel-width-10 mapping.
- Preserve grid, CTA size, shared memory, copy/MMA/epilogue instructions and
  arithmetic.
- Exhaustively prove all 666 coordinate mappings, require boundary bit
  identity, then interleaved S1 and NCU. No local S2 proxy.

## Evidence and prediction

The generic helper retains five runtime `div.u32` operations in PTX because it
reads runtime grid dimensions. For the fixed grid there are four panels with
known sizes: three 18x10 panels and one 18x7 tail panel. Constant division
should remove runtime divides and shorten the address prologue without
changing traversal. Expected S1 gain is 0.2-0.8%, with no resource increase.

## Stop rule

Any coordinate mismatch, output mismatch, spill/register increase, or absent
instruction reduction rejects before whole-graph S2. A positive explained S1
candidate proceeds directly to the natural current S2 graph.
