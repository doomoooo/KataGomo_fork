# Archive contract

This directory may contain compact local dependency archives and manifests. It
must not become a second copy of the multi-gigabyte results tree.

Recognized dependency cache layout:

```text
archive/
  apt/                 # optional .deb files
  wheels/              # optional Python wheelhouse
  git/NAME.bundle      # optional Git bundles
```

Binary/bootstrap acquisition checks these locations before network access.
For source-capable optimizer dependencies, a Git bundle is only a seed: setup
still fetches current upstream HEAD before compiling locally. A packaged
deployment uses its hash-checked wheelhouse and binaries without any source
fetch or rebuild. Large final
optimization artifacts should instead live in persistent data storage and be
listed in a future manifest with:

- logical component and architecture;
- source and producing commit;
- byte size and SHA-256;
- correctness/performance status;
- persistent path;
- relationships to plan entries and generated binaries.

The initial workspace scan found no dependency wheelhouse or source archive.
`/workspace/trainingdata/2026-08-01npzs.tgz` is training data and must not be
misclassified as an environment dependency archive.
