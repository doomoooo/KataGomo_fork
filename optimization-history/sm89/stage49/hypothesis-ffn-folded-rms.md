# Stage 49 hypothesis: fold FFN RMSNorm into dual GEMM

Scope is exact 19x19, B13, S2 on RTX 4090. The accepted control remains the
frozen Stage 46 binary and `/workspace/bench-cuda-gpu0-4090-s2.cfg`.

## Evidence before implementation

The post-Stage-48 full-graph checkpoint ranks the dual FFN GEMM as the largest
exclusive family (53.091 ms, 20.06% of GPU busy time). A three-launch NCU sample
measured the accepted RMSNorm at 4.576 us median and dual GEMM at 41.376 us
median. Each of the 13 FFN blocks has an RMSNorm -> dual projection boundary.

## Single-variable change

Fold each FFN RMS gamma into both projection weight matrices. Replace the full
RMSNorm output with one float inv-RMS per token, and apply that scalar to both
dual-GEMM projection fragments in the SwiGLU epilogue. Attention is unchanged.

Expected local result: one launch remains for inv-RMS, but it writes 18.3 KiB
instead of the 3.44 MiB normalized tensor per FFN block. The dual GEMM reads the
existing residual tensor and gains one scalar load per output row.

Acceptance requires the FFN boundary and full S2 graph to improve consistently.
If the full graph is flat or regresses, reject without instruction-level tuning.
