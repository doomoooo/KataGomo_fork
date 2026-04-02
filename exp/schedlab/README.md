# schedlab

`schedlab` 是一个独立的 `stdexec` 调度实验工程。

它不做真实搜索，但现在已经把 TensorRT backend 单独拆出，并在 TRT 侧走真实
`H2D -> infer -> D2H`。工程主要做三件事：

1. 用 `exec::task`、sender-first 事件和 `single_thread_context` 搭出搜索侧与 TRT 侧的异步结构。
2. 用可调参数、噪音、漂移和突变 schedule 来模拟搜索侧 CPU 成本，并给 TRT 调度器提供可变的时间估计。
3. 用真实 TensorRT plan cache、多 GPU 初始化和多种极端场景验证整个异步调度框架的稳定性。

构建示例：

```bash
cmake -S exp/schedlab -B build/schedlab -DCMAKE_CXX_COMPILER=.local/toolchains/llvm-22.1.1/bin/clang++
cmake --build build/schedlab -j
ctest --test-dir build/schedlab --output-on-failure
```
