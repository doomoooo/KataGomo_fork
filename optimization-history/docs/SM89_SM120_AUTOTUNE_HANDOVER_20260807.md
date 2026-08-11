# SM89 / SM120 跨 batch 自动调优 handover

更新时间：2026-08-07 UTC

接收方：`final-migration` session

## 0. 先看结论

当前工作已经把“逐 candidate 反复生成、configure、编译”的流程改成了“先生成全量 exact-batch AOT，统一 configure/build 一次，再由运行时选择 candidate”的 fat-binary 流程。SM89 的 RTX 4090 B4-B32 discovery 已完成；SM120 的 RTX 5090D 和远端 RTX 5080 都已完成 B4-B32、493 个 AOT entry 的 fat bundle 构建。它们现在都不应再为每个 candidate 重新编译。

尚未完成的是最终性能交付：最新 winner 尚未全部经过长时 whole-graph gate 和 8192-row correctness certification，因此目前没有可以宣称为最终结果的 B4-B32 production tactic plan。任何现有短测峰值都只能用于 discovery，最终报告必须取最新计划、固定二流、至少 1000 timed iterations、至少 2 repeats 的稳定结果。

职责边界已经与用户确认：

- 本 handover 提供两条分支的代码、环境、产物、实测和坑点。
- 整仓合并、SM89/SM120 共用基础代码、FlashAttention/CUTLASS 版本统一、安装脚本和最终提交，由 `final-migration` 负责。
- 不要求 cherry-pick 本分支提交；接收方直接查看对应 worktree 的 working tree。
- 不要在 4090 分支强行引入 SM120 兼容层；它是 SM89 工作流的验证分支。

## 1. 仓库与分支状态

| 目标 | worktree | branch / HEAD | 状态 |
|---|---|---|---|
| SM89 / RTX 4090 | `/workspace/katago-4090` | `4090-opt` / `bd6b8a6a32c5b7742b0eb8f872753c3e4d66e638` | 自动调优代码与结果有未提交修改和新增文件；这是应读取的真实状态 |
| SM120 / RTX 5090D、5080 | `/workspace/katago` | `benchmarknn` / `ed509a1c4b062a618a220a90906f2debe418e4cf` | fat-coordinate、FA4 N96、计划长测等修改尚未提交 |
| 最终迁移 | `/workspace/katago-final-migration` | `final-migration` / `2e96b523c3840f4930ce5ad2f26c0bab62892ebc` | 由接收方维护；本工作未改它 |

SM89 最近优化历史中有价值的提交：

- `bd6b8a6`：使用外部持有的 CUDA streams。
- `077dd1d`：native-half plain QKV。
- `1d3b78d`：split QKV/RoPE。
- `6fd19dc`：dual FFN half2 tanh。
- `7d299d0`：SM89 FlashAttention FP16 accumulation。
- `dd4cb33`：历史 exact-B13 优化。现在脚本已将这些经验拆成所有 batch 都会经历的候选族，B13 不再有流程特判。

SM120 working tree 的核心新增文件：

- `python/sm120_prepare_coordinate_fat.py`
- `python/sm120_coordinate_search.py`
- `python/tests/test_sm120_coordinate_search.py`
- `cpp/neuralnet/fa4_aot/sm120_search_fa4_fat_stub.cpp`
- `python/sm120_prepare_fa4_overlay.py`（实验脚本，见 FA4 警告，不可直接作为最终安装方案）
- `cpp/neuralnet/fa4_aot/flash-attn-both16.patch`（当前是坏的 draft，不可合并）

SM89 接收方必须一起检查的代码集合（不要只复制 Python）：

- 构建/配置：`cpp/CMakeLists.txt`、`cpp/program/setup.cpp`、`cpp/program/setup.h`、二流 baseline config。
- runtime/backend：`cudabackend.cpp`、`cudabackend_sm89.cpp/.h`、`cudabackend_sm89_forward.cpp/.h`、`cudabackend_sm89_kernels.cu/.h`。
- GEMM/FA：`cudabackend_sm89_dual_gemm.cu`、`cudabackend_sm89_linear2_gemm.cu`、`cudabackend_sm89_qkv_rope_gemm.cu/.h`、`cudabackend_sm89_flash.cu/.h`。
- 新 runtime tactic registry：`cudabackend_sm89_tactic_kernels.cpp/.h`。
- benchmark/provenance：`cpp/command/benchmarknn.cpp`、`cpp/neuralnet/cudnnquerymutex.h`。
- Python/doc/test：全部 `python/portable_*.py`、`python/tests/test_portable_tactic_workflow.py`、`docs/portable-tactic-workflow.md`。

两边 `git diff --check` 均通过。SM89 新增 Python 经 `py_compile` 通过；SM120 相关脚本也经 `py_compile` 通过。当前没有本地或远端扫描进程仍在运行。

## 2. 硬件发现：不要硬编码设备序号和 SM 数量

用户明确要求硬件能力通过 CUDA 接口获取。已有脚本会调用 CUDA runtime 查询 device properties；不要用产品名、固定 SM 数、固定 L2 大小或固定 GPU ordinal 做逻辑判断。

当前本地主机的 ordinal 已与历史结果不同：

| 当前 ordinal | GPU | CC | UUID | PCI | 显存 | driver |
|---:|---|---:|---|---|---:|---:|
| 0 | RTX 4090 | 8.9 | `GPU-e40356d3-9eb9-8f64-4fa9-1fd181a77867` | `01:00.0` | 24564 MiB | 595.80 |
| 1 | RTX 5090 D | 12.0 | `GPU-a05bf432-3f6b-83c1-98b1-aec8f8a6fd69` | `21:00.0` | 32607 MiB | 595.80 |
| 2 | RTX 4090 | 8.9 | `GPU-765f1380-bc35-8071-59a0-98df8ca3bc17` | `C1:00.0` | 24564 MiB | 595.80 |

历史 5090D 配置和结果写的是 GPU 2；它在当前机器上是 GPU 1。执行前必须重新发现设备，并生成/覆盖 config 中的 `cudaDeviceToUseThread*`。建议以 UUID + compute capability 作为操作员确认，以 CUDA API 返回的 SM count、L2、shared memory、cluster 能力等作为搜索空间输入。

SM89 对应实现入口是 `/workspace/katago-4090/python/portable_cuda_device.py`。SM120 的空间物化在 `/workspace/katago/python/sm120_tactic_search.py`。SM120 的 active-cluster 估算已改为读取 `cudaDevAttrMultiProcessorCount`，不再写死 5090D SM 数。

远端 5080：

- SSH：`wangyize@10.101.3.156`，hostname `vgpu`。
- GPU 0：RTX 5080，CC 12.0，16303 MiB。
- UUID：`GPU-9b2f97e1-ec8d-23e0-352c-c4dfc41faa48`，PCI `01:00.0`。
- driver：595.84。

## 3. 固定 workload 身份

两条线使用同一个模型：

- 压缩模型：`/workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz`
- SHA256：`1881600caab9e9d85a3dd6a019e9b8e7d2c237b5f984e13ed49a8645be3077c6`
- 解压执行副本：`/workspace/models/b11c768h12nbt3tflrs-fson-silu.bin`
- 解压 SHA256：`38d03bb990f774c0b1676b0a00feee4b05b61ed1a900cfa93a8f123af52e48ae`

SM89 二流 baseline config：

- `/workspace/katago-4090/docs/baseline-configs/bench-cuda-gpu0-4090-s2.cfg`
- SHA256：`d7b738db2cd74fefb04f7545f476d7c4bd3f712b56792809b0486dbfc9df6141`

SM120 历史 5090D 二流 config：

- `/workspace/bench-cuda-gpu2-5090d-s2.cfg`
- SHA256：`ce579bca54cd59743bdf29d55b40f34398acdf6d233f71c8077963323d32f1a1`
- 注意该文件写死了 device 2，只能作为 workload/tactic baseline，不能原样用于当前 ordinal。

计划必须至少绑定：源模型 hash、配置 hash、GPU class/CC、stream count、batch set、search-space hash、生成器/patch/module hash、fat binary hash、长测 evidence hash。接收方环境不必字节级完全一致，但任何不匹配都应显式报告，不能静默当成相同计划。

## 4. 本地主机编译与 Python 环境

系统：Ubuntu 24.04.4，Linux/glibc 2.39。

- NVIDIA driver：595.80。
- CUDA：`/usr/local/cuda`，13.2.86，`nvcc` build `cuda_13.2.r13.2/compiler.37953736_0`。
- cuDNN：9.25.0。
- include：`/usr/include/x86_64-linux-gnu`。
- library：`/usr/lib/x86_64-linux-gnu/libcudnn.so`（亦可解析到 `/lib/x86_64-linux-gnu`）。
- CMake 3.28.3，Ninja generator。
- GCC/G++ 13.3.0。
- system Python 3.12.3。

生成器主要使用 `/workspace/venv`：

| distribution | version |
|---|---|
| `flash-attn-4` | `4.0.0b25`，site-packages 曾手工加入 both16 语义 |
| `nvidia-cutlass-dsl` | 4.7.0 |
| `quack-kernels` | 0.5.3 |
| `torch` | 2.13.0 (`+cu130`) |
| `tilelang` | 0.1.13 |
| `apache-tvm-ffi` | 0.1.12 |
| `triton` | 3.7.1 |
| `numpy` | 2.5.1 |

完整 `pip freeze`、CMake cache、CUDA properties、`nvidia-smi -q` 已写入扫描 JSON 的 `provenance`；例如 SM89：

`/workspace/katago-4090/results/portable-sm89-e2e/history-discovery-incumbent-b19-b32-gpu0-v6-cudnn-mutex.json`

## 5. final-migration 已构建的依赖集合

权威 manifest：

`/workspace/katago-final-migration/.final-migration-env/source-builds/20260807T081318Z/MANIFEST.tsv`

| 组件 | 版本 | resolved commit | wheel SHA256 / 说明 |
|---|---|---|---|
| CUTLASS DSL | 4.7.0 | `dcf215af68a2d08d305076c152a06f201728cd53` | upstream binary |
| Triton | `3.8.0+git5bcfc513` | `5bcfc513ddbbc64f2688dfb15a4d824c56a9649a` | `57efe48b2efcc7ba3c1e4c29447afc91de012f946906ce5c6531f4bdc103f685` |
| Quack | 0.6.3 | `050387bde3d3f03a26c87279bff2df3173640127` | `63412dd22959bc2d6a66297868f044e0da9a4c755c051a3b1643c1b5a3e89e9e` |
| FlashAttention | `0.0.1.dev1+g69e1bcbe7` | `69e1bcbe77c359c84b3a4589e92a7c076e33a202` | `6ba3976f77e2e67bd6f85ca2b68f7f1b85261908e79e986eb419997ddde9213a` |
| TileLang | `0.1.13+cuda.git12dbf3e9` | `12dbf3e9d30d84b5c27d7b8b672c268457f7eb27` | `9152b9eae45c77d59cb8d5f42517954357638a6e9dae1e2b7958c18dc854ba30` |
| apache-tvm-ffi | 0.1.12 | `3050b0a7bd48e04f853027c5fa1f5ab7bc20b856` | `94401b0488761d06be3891efd4d84c63f1c1d9093c0038d22b9c58b0b6a1d6df` |
| cuDNN frontend headers | — | `ec139877e51f17d6b1d7520d9789f34d1c65f77e` | header source |

锁文件有的 source ref 仍写 `HEAD`；可复现包应使用 manifest 中的 resolved commit，不应在未来重新解析移动的 `HEAD`。

通用部署坑点，不是 SM120 独有：

- 当前 Python wheel closure 是 cp312。远端 5080 系统/生成器 Python 是 3.10，不能直接安装 cp312 wheel。需要隔离 Python 3.12，或从相同 resolved commit 重建 cp310 closure。
- 私有 cuDNN 若布局是 `<root>/lib` 而 CMake 默认找 `lib64`，必须显式传 `-DCUDNN_INCLUDE_DIR=...` 和 `-DCUDNN_LIBRARY=...`。SM89/SM120 都会遇到。
- 5080 主机禁止 apt 安装和禁止引入 Ubuntu 24.04 仓库；此前错误污染已回滚。所有内容放 `/data/wangyize` 私有目录。

## 6. 共同的搜索和计划语义

优化单位是固定模型、固定 19x19、固定双流、exact batch 的 whole graph。不能只测孤立 kernel 后直接生成 plan，也不能把各 family 的独立峰值机械拼接。

正确流程：

1. 用 CUDA runtime 查询设备能力并物化 B4-B32 搜索空间。
2. 把历史优化经验展开成每个 exact batch 的小规模 candidate set。
3. 生成所有 AOT source/object，建立 registry，统一 configure/build 一次。
4. 以 accepted historical seed 或显式 fallback/keep/off seed 开始 accumulated coordinate search。
5. 按 family 顺序扫描：固定已接受的前序选择，只改变当前 family；达到改善阈值才更新 incumbent。
6. 可做额外 pass 复查 family 交互，但不做不可承受的全 Cartesian product。
7. discovery 只用于筛选；对最终 joint winner 做长时 whole-graph gate。
8. 对选中算子做 8192-row FP32 replay/correctness certification。
9. 只有长测与 correctness 都通过后才 finalize tactic plan，接收方可验证 hash 后跳过扫描。

当前共同参数语义：

- discovery：100 timed iterations、50 warmup、1 repeat；只属短测。
- winner 接受阈值：`min_improvement_fraction=0.001`，即至少 +0.1%，否则保留 incumbent，抑制噪声锯齿。
- long gate：至少 1000 timed iterations、50 warmup、至少 2 repeats；spread 上限 10%。实际最终报告建议增加 repeats/持续时间，而不是降低门槛。
- correctness：8192 rows，对比 FP32 reference。
- 最终报告：每个 batch 报告 long-stable `nnEval/s`，绝不使用 candidate scan 中间值或仅 100-iteration 的结果。

## 7. SM89 / RTX 4090 详细状态

### 7.1 文档和入口

主文档：`/workspace/katago-4090/docs/portable-tactic-workflow.md`

入口：

- `python/portable_cuda_device.py`：CUDA device properties。
- `python/portable_tactic_workflow.py`：`space`、`generation-plan`、`artifact-bundle`、`scan`、`gate`、`certify`、`plan`、`validate`、`apply`。
- `python/portable_generate_tilelang_aot.py`：TileLang exact-batch AOT。
- `python/portable_prepare_tilelang_fat_scan.py`：集中生成 fat source/registry。
- `python/portable_fat_scan.py`：fat candidate runtime dispatch。

### 7.2 每个 batch 必经的 20 个历史优化 family

B4-B32 全部按相同流程，不允许 B13 特判：

1. `wide_qkv`
2. `wide_ffn`
3. `fused_residual`
4. `rmsnorm`
5. `exact_mask`
6. `qkv_rope`
7. `fa4`（fp32/both16）
8. `dual_ffn`（含 TileLang）
9. `linear2`（含 TileLang）
10. `outproj`
11. `preconv`
12. `postconv_bn`
13. `pointwise`
14. `l2`
15. `initial_conv`
16. `initial_global`
17. `policy_p1`
18. `wide_head`
19. `head_bn`
20. `value_terminal`

Stage68/历史 B13 算子中可跨 batch 成立的部分都已作为 candidate family；真正 exact-B13 shape/schedule 不会盲目复用于其他 batch，而是由 exact-batch AOT 重新生成。SM 数、wave 数、occupancy 等不得从 4090 型号推断，应由 CUDA properties 与编译产物资源信息决定。

### 7.3 SM89 FlashAttention 依赖事实

SM89 也依赖对上游 FlashAttention/CUTLASS 的 KataGo patch，不能说它“无需 patch”：

- 上游 checkout：Dao-AILab/flash-attention commit `5835c733e7e9c07606b045255768e8a7e9e851bd`。
- CUTLASS submodule：`7127592069c2fe01b041e174ba4345ef9b279671`。
- patch：`/workspace/katago-4090/cpp/neuralnet/flash-attention-sm89.patch`。
- patch SHA256：`cb256a2f933797ae5f2cf9cc3dded8a3452ace510bc8aacd399789d2ed305e20`。
- CMake 会检查 patch marker，未应用会拒绝配置。

该 patch 参数化 `ElementAccum`，选择 SM80 family 的 F16 accumulator MMA，做 typed softmax rescale，并开放 tile override。运行时 wrapper `cpp/neuralnet/cudabackend_sm89_flash.cu` 对 both16 使用 `cutlass::half_t` 模板实参。

已验证这份 SM89 patch 可以 clean apply 到 final-migration 的 FlashAttention `69e1bcbe...` 源树，且相关 Hopper target 文件未冲突、CUTLASS pointer 同为 `712759...`。这只证明版本统一有可行性；是否统一由 final-migration 决定。注意 final-migration 当前 FlashAttention 源树没有初始化 submodule，若编 SM89 C++ FA，必须带 CUTLASS submodule。

### 7.4 fat bundle 与 discovery 进度

当前 bundle：

- manifest：`/workspace/katago-4090/results/portable-sm89-e2e/artifact-bundle-b4-b32-v6-cudnn-mutex.json`
- binary：`/workspace/katago-4090/build-sm89-tactic-b4-b32/katago`
- binary SHA256：`0482ee0a4618a93fcc8f5df711d8514d930d9fd597cde5e95afb26beefe7bf15`
- AOT entries：290，其中 `dual_ffn=174`、`linear2=116`。
- manifest 声明 `complete_history_coverage=true`。

最新 discovery 分段：

- B4-B14：`results/portable-sm89-e2e/history-discovery-incumbent-b4-b18-gpu0-v5.json`，1034 rows，实际完成 B4-B14。
- B15-B18：`results/portable-sm89-e2e/history-discovery-incumbent-b15-b18-gpu0-v6-cudnn-mutex.json`，376 rows。
- B19-B32：`results/portable-sm89-e2e/history-discovery-incumbent-b19-b32-gpu0-v6-cudnn-mutex.json`，1316 rows，已完成。

v6 新增 `cpp/neuralnet/cudnnquerymutex.h`：只在模型初始化阶段，对 legacy cuDNN convolution algorithm query 使用 process-global mutex。原因是部分 cuDNN release 在两个 NN server 初始化线程并发 query 时会卡死。它不包 inference hot path，与 GPU lock/持久化 clock 无关。

重要一致性问题：B4-B14 的 v5 扫描早于 v6 mutex/bundle 命名。虽然 mutex 理论上只影响 init，不应改变 steady-state winner，最终交付前仍应使用 v6 同一 binary SHA 重跑 B4-B14 discovery 或至少重跑完整 long gate，避免 evidence/binary 身份不一致。B15-B32 是 v6。

### 7.5 现有 long 结果只能作历史参照

`history-long-gate-b4-b18-gpu0-v4.json` 中有旧 v4 的稳定值：

| batch | nnEval/s | batch | nnEval/s | batch | nnEval/s |
|---:|---:|---:|---:|---:|---:|
| 4 | 2740.518 | 9 | 3318.037 | 14 | 3398.384 |
| 5 | 2906.875 | 10 | 3330.846 | 15 | 3123.766 |
| 6 | 3107.054 | 11 | 3323.963 | 16 | 3363.651 |
| 7 | 3239.013 | 12 | 3422.421 | 17 | 3370.149 |
| 8 | 3255.172 | 13 | 3358.604 | 18 | 3426.245 |

这些数早于最新 v5/v6 discovery 和当前 fat binary，不能作为最终 plan 的 long evidence。B19-B32 尚无 long gate。当前只有 B4 有旧 correctness certificate（`replay-final-b4-v2-vs-fp32.json` 及对应 certified gate）。因此目前没有 B4-B32 production plan。

### 7.6 SM89 命令模板

先查询当前设备，不要沿用历史 ordinal。以下 `<...>` 均需替换：

```bash
cd /workspace/katago-4090

python3 python/portable_tactic_workflow.py space \
  --architecture sm89 --gpu-class rtx4090 --device <DEVICE> \
  --batches 4-32 --streams 2 \
  --output <OUT>/space.json

python3 python/portable_tactic_workflow.py generation-plan \
  --space <OUT>/space.json --phase full \
  --output <OUT>/generation-plan.json

# 依据 generation-plan 用 portable_prepare_tilelang_fat_scan.py
# 分别生成 dual_ffn 与 linear2 manifests，然后只 configure/build 一次。

python3 python/portable_tactic_workflow.py artifact-bundle \
  --space <OUT>/space.json --binary <BUILD>/katago \
  --manifests <DUAL_FFN_MANIFEST> <LINEAR2_MANIFEST> \
  --output <OUT>/artifact-bundle.json

python3 python/portable_tactic_workflow.py scan \
  --space <OUT>/space.json --binary <BUILD>/katago \
  --config docs/baseline-configs/bench-cuda-gpu0-4090-s2.cfg \
  --model /workspace/models/b11c768h12nbt3tflrs-fson-silu.bin \
  --model-identity /workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz \
  --artifact-bundle <OUT>/artifact-bundle.json \
  --device <DEVICE> --streams 2 --batches 4-32 \
  --phase discovery --iterations 100 --warmup 50 --repeats 1 \
  --min-improvement-fraction 0.001 --resume \
  --output <OUT>/discovery.json --raw-dir <OUT>/raw-discovery

python3 python/portable_tactic_workflow.py gate \
  --space <OUT>/space.json --discovery <OUT>/discovery.json \
  --binary <BUILD>/katago \
  --config docs/baseline-configs/bench-cuda-gpu0-4090-s2.cfg \
  --model /workspace/models/b11c768h12nbt3tflrs-fson-silu.bin \
  --model-identity /workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz \
  --artifact-bundle <OUT>/artifact-bundle.json \
  --device <DEVICE> --batches 4-32 \
  --iterations 1000 --warmup 50 --repeats 2 \
  --output <OUT>/long-gate.json --raw-dir <OUT>/raw-long

python3 python/portable_tactic_workflow.py certify \
  --gate <OUT>/long-gate.json \
  --comparison 4=<B4_8192_REPLAY.json> \
  --comparison 5=<B5_8192_REPLAY.json> \
  --output <OUT>/long-gate-certified.json

python3 python/portable_tactic_workflow.py plan \
  --space <OUT>/space.json --results <OUT>/long-gate-certified.json \
  --batches 4-32 --output <OUT>/tactic-plan.json
```

`certify` 需要为 B4-B32 各给一个 `BATCH=PATH`，上面只示意前两个。计划可用 `validate` 检查接收方身份，用 `apply` 生成 exact-batch config overrides。

### 7.7 SM89 测试

```bash
cd /workspace/katago-4090
python3 -m unittest python.tests.test_portable_tactic_workflow
```

结果：12 tests passed。

## 8. SM120 / RTX 5090D、5080 详细状态

### 8.1 文档和入口

主文档：`/workspace/katago/docs/sm120-tilelang-fat-scan.md`

新推荐流程：

- `sm120_tactic_search.py space`：从 CUDA API 物化空间。
- `sm120_prepare_tilelang_fat_scan.py`：生成/复用 FFN、QKV、Linear2 的 TileLang manifests。
- `sm120_prepare_coordinate_fat.py`：把 historical FFN、TileLang/CuTe QKV、Linear2、FA4 全部投影到 B4-B32，统一 configure/build 一次。
- `sm120_coordinate_search.py --fat-bundle`：不生成、不 configure、不 build，只在同一 binary SHA 内切换 runtime candidate。
- `sm120_measure_joint_plan.py --fat-bundle`：同一 fat binary 上长测最终 joint plan。
- `sm120_tactic_plan.py finalize`：附上 long-stability evidence 后才允许 bypass scan。

coordinate families：FFN、QKV、Linear2、FA4、persisting L2。无 seed plan 时使用 deterministic fallback-first seed；它不可部署，但仍会扫描每一个 candidate，不会因为缺历史 plan 而跳过优化点。

FA4 搜索空间对每个 B4-B32 都包含 N64、N96、N128；N96 不再是 B13 特判。用户提到的 N96/B13 约 +0.3% 短测，已作为全局 FA4 dimension 纳入。

### 8.2 本地 5090D fat bundle

完整 bundle 已生成：

- manifest：`/workspace/results/sm120/cross-batch-search/fullflow-5090d-fat-20260807/b4-b32/bundle/manifest.json`
- binary：`/workspace/results/sm120/cross-batch-search/fullflow-5090d-fat-20260807/b4-b32/build/katago`
- SHA256：`19fe0c57478cf49c2561a4668413ea6b6e177f9da291499808aa1f26c23b90ac`
- batch：B4-B32，共 29 个。
- AOT：493 entries：FFN 203、QKV 116、Linear2 87、FA4 87。
- configure：2.006 s。
- build：172.687 s。
- 全流程时间：约 2026-08-07 21:25:50 至 21:36:22 UTC。

manifest 的 `commands.configure` 记录了所有源/object/registry 的完整 CMake argv，不要手抄那几百个路径。其基础形态是：

```bash
cmake -S /workspace/katago/cpp -B <BUILD> \
  -DUSE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release \
  -DKATAGO_CUDA_ARCHITECTURES=120 \
  -DSM120_SEARCH_FFN_FAT_SOURCES=... \
  -DSM120_SEARCH_FFN_FAT_REGISTRY_SOURCE=... \
  -DSM120_SEARCH_QKV_FAT_SOURCES=... \
  -DSM120_SEARCH_QKV_FAT_OBJECTS=... \
  -DSM120_SEARCH_QKV_FAT_REGISTRY_SOURCE=... \
  -DSM120_SEARCH_LINEAR2_FAT_SOURCES=... \
  -DSM120_SEARCH_LINEAR2_FAT_REGISTRY_SOURCE=... \
  -DSM120_SEARCH_FA4_FAT_SOURCES=... \
  -DSM120_SEARCH_FA4_FAT_OBJECTS=... \
  -DSM120_SEARCH_FA4_FAT_REGISTRY_SOURCE=...
cmake --build <BUILD> -j8
```

这个 bundle 生成于最新 `build_aot.py` 增加 FA module source hash metadata 之前；AOT 本身可用，但它不是最终可复现发布包。发布时应重新生成一次，使 manifest 包含 FlashAttention distribution、module path/version/hash。

本地旧 coordinate 文件 `/workspace/results/sm120/cross-batch-search/fullflow-5090d-20260807/coordinate.json` 有 130 rows/23 decisions，但来自旧空间和旧“多次 build”流程，不是上述 v7 B4-B32 fat bundle 的最终 coordinate result。新 fat bundle 的 B4 smoke 曾开始，但只有 6 rows/0 decisions，已中断。因此本地 5090D 仍需跑完整 fat-coordinate 和 long gate。

### 8.3 现有 SM120 吞吐量证据

文档中校正后的二流 whole-graph 短测曲线，仅用于定位峰谷：

| batch | short nnEval/s | batch | short nnEval/s |
|---:|---:|---:|---:|
| 13 | 4265.6 | 19 | 4368.8 |
| 14 | 4332.3 | 20 | 4065.8 |
| 15 | 4306.6 | 25 | 4079.5 |
| 16 | 4251.5 | 27 | 4047.5 |
| 18 | 4263.7 | 32 | 4154.0 |

不要把这些值写入最终报告。文件 `joint-plan-5090d-s2-long-key.json` 虽有 `long` 字样，但仅 100 iterations、3 repeats、且只覆盖 9 个 batch，不符合 >=1000 iteration 的最终门槛。

L2 是真实的 discrete dimension，不是固定开关。短测例子：

- B25：ratio 0.75 约 4164，1.0 约 4104，off 约 4073。
- B27：ratio 1.0 约 4072，off 约 4035。

Nsight whole-graph 资料：`/workspace/results/sm120/cross-batch-search/nsight-joint-5090d-s2/`。

峰值附近资源签名：

- historical TileLang FFN B14/B19/B20：167 regs，32.768 KiB dynamic shared memory，约 3 CTA/SM。
- CuTe packed QKV：288 threads，107 regs，99.328 KiB，1 CTA/SM，历史 cluster Z=170。
- TileLang Linear2：B13/B27 为 162 regs、65.536 KiB、约 3 CTA/SM；B20 为 210 regs、49.152 KiB、约 2 CTA/SM。
- FA4：168 regs，16.384 KiB，约 3 CTA/SM。

当前仍可能遗漏的 search dimensions：register cap、cluster dimensions、active clusters 与整图 cost model 的联动。脚本已经显式生成 wave candidates 并用 CUDA API 查询 SM count，但尚未形成足够强的自动 occupancy/cluster cost model。接收方合并时不要删除这些候选或退回硬编码 5090D magic number。

### 8.4 SM120 FA4 both16：最重要的迁移警告

final-migration 的 clean FlashAttention wheel（`69e1bcbe...`）在 Python CuTe path 中仍把 QK/PV accumulator 写死为 FP32，不能生成已验证的 both16 kernel。本地 `/workspace/venv` 的 `flash-attn-4==4.0.0b25` 曾直接修改 site-packages，因而能生成 accepted both16；这不是可复现安装方式。

accepted 本地模块 hash：

| module | accepted patched SHA256 | final-migration clean SHA256 |
|---|---|---|
| `flash_attn/cute/flash_fwd.py` | `3b97ed4cb20bb867d97fdea7eb03a86ccf32fc4b6f0c8e1ef0302ff1b4728045` | `e9d890e10611ce48dc57c40570a7b65d1ab802c508f785e5536fb10044db65f3` |
| `flash_attn/cute/mask.py` | `93f265c7a02295c4b63027fc8c10fe3a371669f579415e2f3a0480d069a6558b` | `700209e91543f51b11d3bf7020885ef7442136b31dbc9e74b4c668f07d2cc6c7` |
| `flash_attn/cute/softmax.py` | `a09b2a118c26c0547938ffbc302f376076b27aae5689a390bc7e5cfb574c21d2` | `baeefaeed5379fc046180351fe474cdcd6b48fc7f717f2ada8ef950046cdc183` |

需要保留的最小语义：

- QK/PV accumulator 类型参数、typed fragments/MMA。
- dtype-aware FP16 mask tail `-inf`。
- typed online-softmax exp store。
- typed PV rescale。
- `build_aot.py` 中对 `Softmax.rescale_O` 的 FP16 PV monkeypatch 语义，或把等价修复正式落入上游 patch。

性能与精度证据在 `/workspace/results/sm120/stage3/REPORT.md`：固定 S361/H12/D32 FP16 下，isolated FA4 由 16.36 us FP32 降到 11.80 us both16（-27.9%）；双流 Nsys 22.387 降到 17.946 us（-19.8%）；B13 whole graph mean +0.446%、median +0.545%；8192-row precision gates 全通过。

当前 `/workspace/katago/cpp/neuralnet/fa4_aot/flash-attn-both16.patch` 不是可用补丁：

- SHA256：`0813b0f0506faa30587c0f0cd18e406f3ef55fdc1dc4538670c5baedcd1b05b5`。
- 对 clean `69e1bcbe` 源树执行 `patch --dry-run` 报 `malformed patch at line 10`。
- 它还混入了无关的 `softmax.finalize` sink 语义变化，并有重复/碎片化 PV rescale 修改。

因此 final-migration 不应合并这个 draft，也不应使用 `sm120_prepare_fa4_overlay.py` 在安装后修改 site-packages。正确做法是：基于 pin 到 exact commit 的 FlashAttention 源码，整理最小 patch，在构建 wheel 之前应用，记录 patch SHA、patched source hashes、wheel SHA。是否把 SM89/SM120 合到同一 FlashAttention 版本由接收方决定。

`cpp/neuralnet/fa4_aot/build_aot.py` 当前已增加模块身份记录：FlashAttention distribution/version，以及 `flash_fwd`、`flash_fwd_sm120`、`mask`、`softmax` 的路径和 SHA；合并时应保留。

### 8.5 远端 RTX 5080 验证

远端环境：

- OS：Ubuntu 22.04.5。
- CUDA：`/data/wangyize/katago/opt/cuda-13.2`，实际 13.2.78，compiler build `37668154_0`。
- cuDNN：`/data/wangyize/katago/opt/cudnn-9.24.0`，9.24.0。
- CMake 4.1.2，GCC 11.4。
- Python：`/data/wangyize/katago/venvs/flash-attn-sm120/bin/python3`，3.10.12。
- direct venv：FlashAttention dev `gdf61ab6c4`、CUTLASS DSL `4.6.0.dev0`、Quack 0.6.1、Torch 2.13、apache-tvm-ffi 0.1.13.post0、Triton 3.7.1；这个 venv 无 TileLang。
- TileLang 独立 package root：`/data/wangyize/katago/venvs/tilelang/lib/python3.10/site-packages`。
- CuTe packages：`/data/wangyize/katago-b13-sm120/toolchains/cute-dsl-4.7-py310`。
- CUTLASS source：`/data/wangyize/katago-b13-sm120/toolchains/sources/cutlass-e05f953a`，required commit `e05f953a5b3d38adc240df2ff928e0421c2abba3`。
- final-migration runtime 解包：`/data/wangyize/katago-tar-validation-20260807T205459Z/runtime`。
- 模型：`/data/wangyize/katago/models/b11c768h12nbt3tflrs-fson-silu.bin.gz`。
- config：`/data/wangyize/katago-b13-sm120/bench-cuda-b13-s2.cfg`。

测试 repo 完全隔离在 `/data/wangyize/katago-fat-scan-test-20260807`，未改系统仓库。临时 FA overlay 位于其 `results/5080-b4-smoke/python-overlay`：它从 final-migration wheels 展开后，手工复制 accepted `flash_fwd/mask/softmax`，保留 `.deployed-wheel` backups。这只能作为兼容性验证，不能作为发布方法。

B4 smoke：

- 17 linked AOT entries。
- configure 0.075 s，build 35.014 s。
- binary SHA256：`fc101391f45f6043f0466c8ac1a028de9a0cc425afef079a359cda34d2286319`。
- coordinate：28 rows、5 decisions，同一 binary SHA，0 次 candidate rebuild。
- short winners：FFN historical tanh_half2 1487.401；QKV TileLang m128n128k32s3 1601.035；Linear2 fallback 1602.908；FA4 N128 2024.480；L2 ratio 1.0 2026.572。
- medium samples：2025.526 / 2005.728 / 2014.472，median 2014.472，relative spread 0.00983。
- 这些仍是 short/medium discovery，不是最终 long。

远端 B4-B32 fat bundle 已完成：

- manifest：`/data/wangyize/katago-fat-scan-test-20260807/results/5080-b4-32-fat-20260807/coordinate-fat/manifest.json`
- binary：`/data/wangyize/katago-fat-scan-test-20260807/results/5080-b4-32-fat-20260807/build-coordinate-fat/katago`
- SHA256：`342177f75d6a60cda17ab69fd6a4e3bf7c88d7156a12d61ccb1791993da9ca51`
- 493 entries：FFN 203、QKV 116、Linear2 87、FA4 87；B4-B32 全覆盖。
- configure 1.364 s，build 149.973 s。
- 约 22:00:38 至 22:08:10 UTC 完成。

尚未在远端运行完整 B4-B32 coordinate scan。下一步可以直接复用该 manifest，不需要重新生成/编译。

### 8.6 SM120 命令模板

先物化空间：

```bash
cd /workspace/katago

python3 python/sm120_tactic_search.py space \
  --gpu-class rtx5090d --device <DEVICE> \
  --batches 4-32 --streams 2 \
  --output <OUT>/space.json
```

先用 `sm120_prepare_tilelang_fat_scan.py` 为 FFN/QKV/Linear2 生成三个 manifest，然后一次性准备 whole-coordinate binary：

```bash
/workspace/venv/bin/python3 python/sm120_prepare_coordinate_fat.py \
  --repo /workspace/katago \
  --space <OUT>/space.json --batches 4-32 --device <DEVICE> \
  --output-dir <OUT>/bundle --build-dir <OUT>/build --jobs 8 \
  --generator-python /workspace/venv/bin/python3 \
  --fa4-python /workspace/venv/bin/python3 \
  --cutlass-root <PINNED_CUTLASS_ROOT> \
  --tilelang-ffn-manifest <FFN_MANIFEST> \
  --tilelang-qkv-manifest <QKV_MANIFEST> \
  --tilelang-linear2-manifest <LINEAR2_MANIFEST>
```

私有 CUDA/cuDNN 环境追加重复的 `--cmake-arg=-DNAME=VALUE`，至少：

```text
--cmake-arg=-DCMAKE_CUDA_COMPILER=<CUDA>/bin/nvcc
--cmake-arg=-DCUDNN_INCLUDE_DIR=<CUDNN>/include
--cmake-arg=-DCUDNN_LIBRARY=<CUDNN>/lib/libcudnn.so
```

同一 fat binary 上运行 coordinate：

```bash
python3 python/sm120_coordinate_search.py \
  --space <OUT>/space.json \
  --fat-bundle <OUT>/bundle/manifest.json \
  --output <OUT>/coordinate.json \
  --plan-output <OUT>/selected-plan-short.json \
  --config <CONFIG_WITH_CURRENT_DEVICE> \
  --model /workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz \
  --device <DEVICE> --batches 4-32 --streams 2 \
  --passes 1 --iterations 100 --warmup 50 --repeats 1 \
  --min-improvement-fraction 0.001
```

无 `--seed-plan` 会用 fallback-first seed 并扫描所有 candidate。短 plan 故意不可 bypass。

长时 joint gate：

```bash
python3 python/sm120_measure_joint_plan.py \
  --plan <OUT>/selected-plan-short.json \
  --space <OUT>/space.json \
  --fat-bundle <OUT>/bundle/manifest.json \
  --output <OUT>/joint-long.json \
  --config <CONFIG_WITH_CURRENT_DEVICE> \
  --model /workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz \
  --device <DEVICE> --batches 4-32 --streams 2 \
  --iterations 1000 --warmup 50 --repeats 2

python3 python/sm120_tactic_plan.py finalize \
  --plan <OUT>/selected-plan-short.json \
  --joint-result <OUT>/joint-long.json \
  --space <OUT>/space.json \
  --model /workspace/models/b11c768h12nbt3tflrs-fson-silu.bin.gz \
  --config <CONFIG_WITH_CURRENT_DEVICE> \
  --batches 4-32 --streams 2 \
  --output <OUT>/tactic-plan-final.json
```

原则是只能将与同一 space/model/config/binary 匹配的 `joint-long.json` 附到 short plan。接收方合并后若 CLI 有变化，应以合并后的 `--help` 为准，但不能削弱身份或长测校验。

### 8.7 SM120 测试

```bash
cd /workspace/katago
python3 -m unittest discover -s python/tests -p 'test_sm120*.py'
```

结果：39 tests passed。

## 9. final-migration 建议执行顺序

1. 从 `/workspace/katago-4090` 合入 portable SM89 workflow、runtime dispatch、CUDA capability discovery、20-family candidate history、cuDNN query mutex和测试；不要加入 SM120 兼容特判。
2. 从 `/workspace/katago` 合入 SM120 fat registries、`prepare_coordinate_fat`、fat-coordinate、fat joint gate、FA4 N64/N96/N128 全 batch 空间、硬件属性查询和测试。
3. 整理 FlashAttention：以 exact commit 构建，分别保留 SM89 C++ patch 和 SM120 Python CuTe both16 的最小可复现 patch；不要合入当前坏 draft，不要依赖 site-packages overlay。
4. 统一安装脚本时解决 cp312/cp310 ABI 和私有 cuDNN `lib` 路径问题；不得修改远端 5080 系统 apt/repo。
5. 重新生成最终 fat bundles，使 manifest 包含依赖 resolved commits、wheel/module/patch/source/object/registry/binary hashes和完整 CMake argv。
6. 本地 4090：至少用同一 v6/final binary 重跑 B4-B14 一致性，再对 B4-B32 做 long gate + correctness。
7. 本地 5090D：使用当前 493-entry bundle 或重生成的 final bundle跑 B4-B32 coordinate，再 long gate + correctness。
8. 远端 5080：可直接从 SHA `342177...` bundle 开始 coordinate 验证；发布前仍用最终依赖 closure 重生成一次。
9. 分设备生成 tactic plan。plan 是可分发的扫描结果，不是跨 GPU 型号万能配置；接收方 validate 后可以免扫描。
10. 最终报告只列 long-stable `nnEval/s`、spread、迭代数、repeat 数、binary/plan hash 和精度 gate；短测值只放 discovery appendix。

## 10. 当前明确未完成项

- SM89 最新 B4-B32 joint winner 的统一 long gate。
- SM89 B5-B32 8192-row correctness certification。
- SM120 5090D 新 fat bundle 的完整 B4-B32 coordinate scan。
- SM120 5090D/5080 新计划的 >=1000 iteration long gate。
- SM120 新计划的 8192-row correctness certification。
- 可用、最小、可 dry-run apply 的 SM120 FlashAttention both16 source patch。
- 最终依赖 closure 的 Python ABI 选择与远端部署验证。
- 基于 NCU/NSYS 的 register cap / cluster / active-wave 更完整搜索维度。

在这些完成前，不应生成标记为 `production_ready` 的 B4-B32 plan，也不应宣称任何短测是“最高长时稳定 nnEval/s”。
