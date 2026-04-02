# cuda_completion_bench results

这份结果只保留重新校准后的有效数据。

之前那组 `~2.22ms` 的结果已经作废。重新排查后确认：

- 当时不该信任那组数字。
- 现在 benchmark 已增加：
  - CUDA device ordinal 到 PCI bus 的打印
  - `already-ready event` sanity test
  - host 侧 breakdown
- 重新在干净设备上测试后，completion tax 回到了合理的微秒级。

## Test setup

- 设备：`CUDA device 1`
- 实际卡：`NVIDIA GeForce RTX 4090`
- PCI bus：`0000:E1:00.0`
- 命令：
  `build/cuda_completion_bench/cuda_completion_bench --device 1 --warmup 100 --iterations 2000 --kernel-us 0,10,100,1000 --schemes blocking,waiter,query,callback`

## Sanity

这一步专门验证“同步 API 自身”是不是已经很重。

- `cudaEventSynchronize(ready)`: `avg 0.36us`, `p95 0.39us`
- `cudaEventQuery(ready)`: `avg 0.12us`, `p95 0.13us`

这说明同步 API 本身不是毫秒级，也不是几十微秒级。

## Main results

### `kernel_us = 0`

| scheme | avg total_us | avg gpu_us | avg tax_us | p95 tax_us | p99 tax_us |
|---|---:|---:|---:|---:|---:|
| `cudaEventSynchronize(caller)` | 6.55 | 2.52 | 4.03 | 4.49 | 4.63 |
| `cudaEventSynchronize(waiter_thread)` | 13.92 | 2.54 | 11.38 | 12.12 | 12.75 |
| `cudaEventQuery(busy_poll)` | 6.52 | 2.45 | 4.07 | 4.50 | 4.61 |
| `cudaLaunchHostFunc` | 20.30 | 2.78 | 17.52 | 19.06 | 23.58 |

### `kernel_us = 10`

| scheme | avg total_us | avg gpu_us | avg tax_us | p95 tax_us | p99 tax_us |
|---|---:|---:|---:|---:|---:|
| `cudaEventSynchronize(caller)` | 16.49 | 12.41 | 4.08 | 4.43 | 4.85 |
| `cudaEventSynchronize(waiter_thread)` | 19.51 | 12.40 | 7.11 | 7.78 | 9.77 |
| `cudaEventQuery(busy_poll)` | 16.40 | 12.40 | 4.01 | 4.27 | 4.40 |
| `cudaLaunchHostFunc` | 36.78 | 12.53 | 24.25 | 28.43 | 32.77 |

### `kernel_us = 100`

| scheme | avg total_us | avg gpu_us | avg tax_us | p95 tax_us | p99 tax_us |
|---|---:|---:|---:|---:|---:|
| `cudaEventSynchronize(caller)` | 101.02 | 97.04 | 3.98 | 4.31 | 4.58 |
| `cudaEventSynchronize(waiter_thread)` | 102.80 | 95.61 | 7.20 | 8.97 | 11.94 |
| `cudaEventQuery(busy_poll)` | 99.42 | 95.46 | 3.96 | 4.29 | 4.39 |
| `cudaLaunchHostFunc` | 119.32 | 95.52 | 23.80 | 25.71 | 30.01 |

### `kernel_us = 1000`

| scheme | avg total_us | avg gpu_us | avg tax_us | p95 tax_us | p99 tax_us |
|---|---:|---:|---:|---:|---:|
| `cudaEventSynchronize(caller)` | 938.36 | 934.16 | 4.19 | 4.65 | 8.29 |
| `cudaEventSynchronize(waiter_thread)` | 942.44 | 934.22 | 8.22 | 10.14 | 15.42 |
| `cudaEventQuery(busy_poll)` | 938.81 | 934.47 | 4.34 | 5.36 | 9.18 |
| `cudaLaunchHostFunc` | 963.02 | 934.89 | 28.12 | 33.60 | 42.11 |

## Takeaways

- `cudaEventSynchronize(caller)` 的额外税大约是 `4us`。
- `cudaEventQuery(busy_poll)` 和 caller-side sync 非常接近，也是 `4us` 左右。
- 专门 waiter 线程会把税抬到大约 `7us ~ 11us`。
- `cudaLaunchHostFunc` 明显更重，平均大约 `18us ~ 28us`，尾延迟也更差。

## Raw output

- [results_device1_clean.txt](/home/wangyize/.katago/KataGomo_fork/exp/cuda_completion_bench/results_device1_clean.txt)
