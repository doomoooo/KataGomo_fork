# SM120 exact-batch TileLang fat scan

The fat-scan path trades a larger one-time compile/link for cheap whole-graph
measurements. It generates every explicitly selected TileLang `(batch, tactic
ID)` translation unit first, links the complete family once, and dispatches by
the exact runtime batch and requested tactic ID. It does not infer an anchor or
restrict generation to the left edge of a throughput plateau.

The default build remains unchanged in behavior: each family has an empty fat
registry and the existing single-candidate search-slot ABI remains available.
Official/library fallback candidates also remain available. Fat entries are
explicit-only and are never selected by `auto`.

## One-command full B1-B32 family scan

Generate a schema-2 space containing all batches first. For example:

```sh
python3 python/sm120_tactic_search.py space \
  --gpu-class rtx5090d --batches 1-32 --streams 2 \
  --output results/space-5090d-b1-b32-s2.json
```

Then add `--fat-scan` to the normal runner. The remaining benchmark arguments
are the same as the single-slot workflow:

```sh
python3 python/sm120_run_tactic_search.py \
  --fat-scan \
  --space results/space-5090d-b1-b32-s2.json \
  --family ffn \
  --repo . \
  --build-dir build/fat-ffn \
  --active-source-dir build/fat-ffn/active \
  --config docs/baseline-configs/bench-cuda-gpu2-5090d-s2.cfg \
  --model /workspace/models/model.bin.gz \
  --device 2 --batches 1-32 --streams 2 \
  --iterations 80 --warmup 15 --repeats 1 \
  --output results/ffn-fat-b1-b32.json
```

`--candidate-ids id-a,id-b` is an optional global filter. Without it, every
TileLang implementation in the selected family's materialized B1-B32 space is
included. Fallback and non-TileLang implementations are not put in the fat
bundle; the runner still measures fallback and records unsupported generators
as before. `--reuse-fat-generated` reuses a prior TU only when its source,
metadata, search-space, and generator hashes all match.

Generation still invokes TileLang once per exact shape because each kernel is
specialized for `batch * 361` tokens. The speedup is that CMake configuration,
the full KataGo link, and binary hashing happen once for the whole family,
instead of once per candidate.

## Prepare and build separately

The bundle can also be prepared without starting whole-graph measurements:

```sh
python3 python/sm120_prepare_tilelang_fat_scan.py \
  --space results/space-5090d-b1-b32-s2.json \
  --family ffn --batches 1-32 --device 2 \
  --output-dir results/generated/ffn-fat
```

The emitted `manifest.json` records all exact keys, hashes, source paths, and
the generated registry. Configure a build manually with:

```sh
cmake -S cpp -B build/fat-ffn -DUSE_BACKEND=CUDA \
  -DSM120_SEARCH_FFN_FAT_REGISTRY_SOURCE=/abs/path/sm120_search_ffn_fat_registry.cu \
  '-DSM120_SEARCH_FFN_FAT_SOURCES=/abs/path/a.cu;/abs/path/b.cu'
cmake --build build/fat-ffn -j4
```

Only one family should be fat-linked for a low-cost scan. The runner resets all
other family fat registries and legacy active slots to stubs, avoiding stale
CMake-cache state. Acceptance remains natural whole-graph S2 total throughput;
the fat mechanism does not add homogeneous or mixed local-S2 gates.

## Link safety

Each generated TU receives a deterministic symbol token derived from family,
exact batch, and the SHA-256 of the tactic ID. Both the CUDA kernel and launcher
use that token. TileLang's header-defined debug helpers (`PrintTraits`,
`debug_print_*`, and `device_assert*`) are macro-renamed with the same token,
which prevents the duplicate linker definitions seen when ordinary generated
TUs are linked together.

Run the CPU/static regression tests with:

```sh
python3 -m unittest python/tests/test_sm120_fat_scan.py
```

