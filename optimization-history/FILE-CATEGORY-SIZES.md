# Source-archive file-category inventory

Inventory captured after consolidation on 2026-08-11. Sizes cover `sm89/`,
`sm120/`, `shared/`, and `research/`; filesystem rounding may make `du` totals
slightly different.

This Git mirror contains only Markdown reports/history and report-like valid
JSON. The other categories describe the external `/workspace/results` evidence
archive and are not tracked here.

| Category | Files | Size | Git policy |
| --- | ---: | ---: | --- |
| Profiler traces/databases (`*.ncu-rep`, `*.nsys-rep`, SQLite, QDSTRM) | 943 | 5.2 GiB | Do not put in Git; keep in the external evidence archive |
| Binaries, models, engines, objects, and build trees | 3,285 | 4.2 GiB | Do not put in Git; reproducible or retained only as raw evidence |
| Generated IR/disassembly (SASS, PTX, cubin-adjacent dumps) | 150 | 425 MiB | Keep externally; commit only small excerpts needed by a report |
| Tabular samples/exports (`*.csv`, `*.tsv`) | 995 | 155 MiB | Commit compact summaries only; keep full traces externally |
| Raw text logs (`*.raw`, `*.err`, `*.out`, `*.log`, large `*.txt`) | 14,813 | 104 MiB | Keep externally; commit only decision-bearing logs when no summary exists |
| Reviewable docs and structured results (Markdown, JSON, YAML, DOT) | 3,117 | 51 MiB | Generally Git-suitable after excluding generated candidate matrices |
| Source and experiment scripts | 1,669 | 18 MiB | Git-suitable when not duplicated from a build tree |
| Rendered documents | 1 | 3.5 MiB | Optional; prefer source Markdown/data |
| Other | 394 | 5.6 MiB | Review case by case |

## Architecture breakdown

| Category | SM89 | SM120 |
| --- | ---: | ---: |
| Profiler traces/databases | 3.1 GiB | 2.0 GiB |
| Binaries/models/build artifacts | 700 MiB | 2.4 GiB |
| Generated IR/disassembly | 404 MiB | 21 MiB |
| Tabular samples/exports | 8.7 MiB | 147 MiB |
| Raw text logs | 53 MiB | 46 MiB |
| Reviewable docs/structured results | 1.1 MiB | 45 MiB |
| Source and scripts | 420 KiB | 18 MiB |
| Rendered documents | 0 | 3.5 MiB |
| Other | 85 KiB | 5.5 MiB |

The remaining binary/model total in the overall table is primarily the 1.2 GiB
`research/luminal/` feasibility tree. Shared baselines and accuracy reports are
about 52 MiB, dominated by one baseline profiler capture.

## Tracked Git subset

Per project policy, Git tracks 265 Markdown histories/reports and 299 valid,
report-like JSON files: 564 files with 12,182,439 content bytes (about 11.6 MiB;
about 19 MiB on disk). Included JSON names or paths identify summaries,
reports, analyses, decisions, comparisons, accuracy/correctness/precision
results, replay results, interference/attribution, or metrics. Search spaces,
historical plans, manifests, coordinates, individual benchmark captures, and
837 text/empty files incorrectly carrying a `.json` suffix remain only in the
external archive.

The remaining profiler databases, full GPU-trace CSV files, old KataGo
binaries, ONNX/TRT engines, object files, and generated SASS should be packed
as a versioned external archive with SHA-256 manifests. Git LFS is possible,
but a release/object-store artifact is preferable for this roughly 10 GiB raw
evidence set.
