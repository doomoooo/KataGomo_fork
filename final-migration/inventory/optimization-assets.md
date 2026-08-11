# Optimization asset map

Read-only observation captured 2026-08-07. Nothing here opens a freeze gate.

## RTX 4090 / SM89

Primary history: `/workspace/results/sm89/HISTORY.md`.

The latest recorded stage is Stage 72. Its M64xN128 retile for the already
accepted both16 FlashAttention was rejected: against M64xN96 it was 2.817%
slower, increased registers from 117 to 168, reduced resident CTAs from four to
three, and reduced achieved occupancy. It made no source change.

The accepted lineage needed for later migration is therefore:

| Concern | Stage | Commit/evidence | Status |
| --- | ---: | --- | --- |
| exact-B13 SM89 base bundle | 1–57 | `dd4cb335`, prior commits | accepted |
| both16 FlashAttention | 59 | `7d299d0` | accepted; M64xN96 remains selected |
| half2-tanh dual FFN | 62 | `6fd19dc` | accepted |
| optional split QKV/RoPE | 64–65 | `1d3b78d`, `077dd1d` | retained but default-off |
| externally owned CUDA streams | 68 | `bd6b8a6` | accepted interface correction |
| graph/eager GTP audit | 71 | report commits `7e6ec01`, `67a4d79` in separate worktree | eager selected for deployment |
| both16 M64xN128 retile | 72 | `stage72/final-decision-summary.json` | rejected, no source change |

Stage 68 is the latest accepted commit in `/workspace/katago-4090`. It reports
that all forward kernels, runtime copies, handles, events, and synchronization
use four explicitly owned non-blocking streams for S2; no forward kernel was on
the default stream. Three 100-iteration measurements were
3,462.633/3,460.038/3,450.634 aligned nnEval/s (mean 3,457.768).

Stage 71 later measured the GTP path with a common-wall physical-B13 metric.
Eager beat the current graph implementation by 1.735% physical rows/s. The
report attributes this to graph-imposed resource-contention phase rather than
submission savings: graph raised overlap while slowing several dominant
kernels. It also records that stream priorities and
`CUDA_DEVICE_MAX_CONNECTIONS=1` are harmful. These conclusions constrain the
future scheduler but are not a frozen plan schema.

## Unfrozen scanner work

`/workspace/katago-4090` is currently dirty with new portable tactic workflow
sources, including `python/portable_fat_scan.py`, workflow/generator helpers,
SM89 tactic kernels, and tests. These files are owned by the scanner session and
must not be copied, formatted, staged, or interpreted as stable. Phase 2 waits
for the complete freeze tuple in `FREEZE-GATES.md`.

## SM120

Primary history: `/workspace/results/sm120/HISTORY.md`; recent evidence lives
under Stages 55–57 and `cross-batch-search/`. These are observation-only until
their owner freezes source, schema, assets, and validation commands.
