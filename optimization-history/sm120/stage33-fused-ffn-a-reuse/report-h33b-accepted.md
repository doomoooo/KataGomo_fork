# H33b accepted: fused-FFN A-fragment reuse

## Decision

Accepted for the fixed RTX 5090 D, B13, 19x19, two-stream target. The old
homogeneous-S2 rejection was a false negative: it reported a `0.275%`
regression, while both real whole-graph screens measured a stable improvement.
Homogeneous and synthetic mixed S2 measurements are now historical only and
have no decision weight.

## Evidence

| Gate | Result |
|---|---|
| Boundary correctness | bit-exact for all 1,802,112 FP16 outputs |
| Isolated S1 | `37.680 -> 34.921 us` (`-7.321%`) |
| Resource mechanism | registers/thread `146 -> 136`, zero spills; grid, threads and dynamic shared memory unchanged |
| Short real S2 graph | `3845.895 -> 3880.935 nn/s` (`+0.911%`), 4/4 adjacent pairs positive |
| Long real S2 graph | `3824.934 -> 3859.725 nn/s` (`+0.910%`), 4/4 adjacent pairs positive |
| Full replay | byte-identical, SHA256 `ed0ed80848d752bc6d64995e91f9bada55c059b5e55ac5bcccb13bf28a3e1a02` |

The long order was `A-B-B-A-B-A-A-B`, 1000 timed iterations and 30 warmups
per arm. Adjacent candidate improvements were `+0.972%`, `+0.891%`,
`+0.636%`, and `+1.142%`.

## Accepted state and profile

`cudaUseFusedFFNAReuseSm120 = true` is enabled only in the fixed target
configuration. The runtime option remains default-false and retains the exact
previous kernel as fallback. Binary SHA256 is
`c60db86740a49516b6aae1067a2bd72df62e8a1b1da7a5f20ccc7ccb559e3091`;
accepted config SHA256 is
`807063182b342fdbdab2dbfe332d4af79b51a8cb80d58b7682a6060614f5f43d`.

A fresh accepted-state profile was collected before choosing another target:

- S2 and matching S1 Nsys, 30 forwards per stream.
- S1 NCU over one complete, contiguous 344-ordinal forward.
- Joined ranking: `stage35-post-a-reuse-profile/accepted-fullgraph-ranking.md`.

The rerank keeps linear2 as the largest interference target: `1058.1 us` work
and `370.3 us` excess per stream-forward. Fused FFN remains the largest work
bucket (`1502.7 us`) but its NCU register count is now 136 as intended.

## Artifacts

- `long-h33b/summary.json`: long whole-graph summary, SHA256
  `4b3a73849134e56efc1165bc40f1524d0c085f5e779259ee926eda4d41c56bcf`
- `accuracy-h33b/`: 8,192-row replay and byte comparison
- `stage35-post-a-reuse-profile/nsys/`: accepted S2/S1 traces
- `stage35-post-a-reuse-profile/ncu/accepted-s1-full-forward.ncu-rep`: full NCU
- `stage35-post-a-reuse-profile/accepted-fullgraph-ranking.{md,json}`: rerank
