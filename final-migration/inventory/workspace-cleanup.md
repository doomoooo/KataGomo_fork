# Workspace cleanup record

Cleanup on 2026-08-07 was intentionally limited to comparison outputs and
reproducible local artifacts. Active optimization worktrees, reports, source
trees, models, offline FP32 inputs, current wheels, and the final distribution
were preserved.

- Deleted all 208 `*.krnn` files below `/workspace`: 38,919,465,627 bytes
  (36.247 GiB). No process had any of those files open immediately before
  deletion. They are not directly recoverable, but are reproducible from their
  corresponding comparison runs.
- Deleted the final-migration local install-smoke prefix after it passed, the
  superseded `20260807T091315Z` bundle/tar, compiler caches, smoke-build output,
  and two non-current source-build directories.
- Preserved `20260807T205459Z.tar`, its external checksums/installer, its
  extracted current bundle, the current `20260807T081318Z` source-wheel build,
  the migration venv, and all acquired upstream repositories.

After cleanup `/workspace` contained 49,041,309,696 bytes of visible files and
zero `*.krnn` files. The filesystem reported 115 GiB available.
