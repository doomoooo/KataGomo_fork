# cuda_completion_bench

一个独立的小 benchmark，用真实 CUDA 测 host 侧 completion 方案的额外税。

当前比较 4 种方案：

- `cudaEventSynchronize(caller)`
- `cudaEventSynchronize(waiter_thread)`
- `cudaEventQuery(busy_poll)`
- `cudaLaunchHostFunc`

统计口径：

- `total_us`
  从 GPU 工作提交完毕后，到 host 侧收到 completion 为止的总时间
- `gpu_us`
  用 `cudaEventElapsedTime` 量出的设备侧时间
- `tax_us = total_us - gpu_us`
  作为这条 completion 路径的近似 host tax
