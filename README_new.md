# KataGomo TensorRT Workflow

This repository includes a small set of helper scripts for building, benchmarking, and running the TensorRT backend.

## 1. Prepare `env.sh`

Copy `env_sample.sh` to `env.sh` and fill in the local paths:

```bash
cp env_sample.sh env.sh
```

Edit at least these variables in `env.sh`:
- `TENSORRT_ROOT`
- `KATAGO_BIN_PATH`
- `KATAGO_MODEL_PATH`
- `KATAGO_CONFIG_PATH`

`env.sh` is shared by all new helper scripts:
- `build.sh`
- `run.sh`
- `python/benchmark.py`

## 2. Build

Run the build script from the repository root:

```bash
./build.sh
```

This script compiles the TensorRT backend and deploys the built `katago` binary to `KATAGO_BIN_PATH`.

## 3. Find the Best Batch Size

Run:

```bash
python python/benchmark.py
```

The script reads defaults from `env.sh`, generates TensorRT plans, benchmarks different batch sizes and CUDA stream counts, and writes results under `benchmark/`.

If you want PNG plots, install `matplotlib` first. Without it, the script can still run, but plotting will be skipped.

## 4. Run Benchmark or GTP

Run a benchmark once:

```bash
./run.sh --benchmark
```

Run GTP mode:

```bash
./run.sh
```

### Precision selection

TensorRT now supports an explicit BF16 precision toggle in addition to the existing FP16 path.

- Config file override: `trtUseBF16=true`
- Helper script flag: `./run.sh --trt-use-bf16`
- Do not enable `trtUseBF16=true` together with `useFP16=true`

This is useful for ONNX models that are unstable in FP16 but run correctly in BF16. For example, on this machine
`b24tf.onnx` reproduces non-finite outputs in FP16 and runs successfully in BF16.

You can inspect all optional runtime overrides with:

```bash
./run.sh --help
```
