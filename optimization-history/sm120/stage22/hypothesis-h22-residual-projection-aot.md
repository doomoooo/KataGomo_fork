# H22: fixed-B13 linear2/out-projection residual AOT

## Evidence

- The current linear2 C1152->C384 and attention out-projection C384->C384
  already use `cublasHgemm(beta=1)` to remove standalone residual kernels.
- The 5080 history retained fixed CUTLASS AOT kernels for both boundaries,
  measuring +2.294% and +1.187% whole-network gains in that regime.

## Mechanism

Use shape-specialized TileLang FP16-accumulator GEMMs at M=4693,N=384 with
K=1152 or K=384. Their epilogues add the existing C384 residual and write it
back in place. This preserves the current launch count while replacing the
generic cuBLAS tile and epilogue.

## Expected change

Each direct boundary must beat `torch.addmm` under two concurrent streams.
Integrate linear2 first, then out-projection as a separate variable. Each must
improve whole B13/S2 throughput by at least 0.5%.

## Risks and gates

- FP16 reduction and residual-add ordering can change the accepted envelope.
- High shared-memory tiles can interfere with the fused FFN and wide QKV.
- Isolated output must pass `rtol=2e-2, atol=2e-2`; unsupported shapes fall
  back; accepted candidates require ordered whole-network A/B, Nsys, and the
  8,192-row accuracy matrix.
