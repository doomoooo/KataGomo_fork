# Profile Details (as of 2026-03-04 UTC)

## 1. 目标与范围

本文件记录当前阶段已经确认的性能事实、代码实现细节和可执行的参数选择方法，覆盖：

- `numSearchThreads`
- `numNNServerThreadsPerModel` (即 cudaStream 数量)
- `nnMaxBatchSize`

聚焦于搜索线程侧统计与 TensorRT 理论性能拼接，不包含新的实机重跑结果。

---

## 2. 已使用的数据与文件

- 理论推理性能（`python/benchmark.py` / trtexec 输出）
  - `build/trtexec_benchmark.b18tf.json`
  - `build/trtexec_benchmark_b28.json`
  - 对应 PNG 图：`build/trtexec_benchmark.b18tf.png`、`build/trtexec_benchmark_b28.png`

- 搜索线程原始状态观测（TSV）
  - `benchmark/search_thread_raw_stats_t16_s4_b4_v100000_20260303_034919.tsv`

- 基准运行脚本和配置
  - `benchmark.sh`
  - `run.sh`
  - `cpp/configs/gtp_example.cfg`

- 相关核心代码
  - 搜索线程统计采集与写盘：`cpp/search/search.cpp`, `cpp/search/search.h`
  - 参数定义：`cpp/search/searchparams.h`, `cpp/search/searchparams.cpp`
  - 配置读取：`cpp/program/setup.cpp`
  - benchmark Elo 经验公式：`cpp/program/playutils.cpp`

---

## 3. 搜索线程原始统计字段（当前版本）

TSV 表头：

- `search_id`
- `root_turn`
- `thread_idx`
- `attempt_idx`
- `root_visits_start`
- `root_visits_end`
- `upper_bound_visits_left`
- `max_depth`
- `should_count_playout`
- `descend_returned`
- `outcome_code`
- `outcome`
- `is_pondering`
- `total_time_ns`
- `gpu_wait_time_ns`

派生关键量：

- `work_time_ns = total_time_ns - gpu_wait_time_ns`（纯 CPU 搜索侧近似）
- `gpu_wait_ratio = gpu_wait_time_ns / total_time_ns`

---

## 4. 当前 TSV 的关键事实（t16/s4/b4/v100000）

### 4.1 数据规模

- 总行数：`1,024,650`（包含表头则 `1,024,651` 行）
- `thread_idx` 覆盖：`0..15`，每线程约 `63k~65k` 行
- `search_id`：`10` 个主要搜索（外加极少初始化行）
- `root_turn`：11 个值（0 只出现极少量初始化样本）

### 4.2 outcome 分布

- `new_node_evaluated`：570,903（约 55.7%）
- `edge_catchup_existing_child`：393,477（约 38.4%）
- `edge_catchup_new_child`：35,775（约 3.5%）
- `new_child_race`：24,359（约 2.4%）
- `init_node_nn_fail`：136（可忽略）

### 4.3 总时间与 GPU 等待占比

- 所有 attempt 的总体：
  - `total_time` 合计约 `1784.21s`
  - `gpu_wait_time` 合计约 `1681.95s`
  - 全局 `gpu_wait_ratio` 约 `94.27%`

- `total_time` 分位数（ms）：
  - p50 `2.843`
  - p90 `3.282`
  - p95 `3.378`
  - p99 `3.676`
  - p99.9 `5.219`

- 高尾 mostly 来自 GPU wait：
  - `total_time >= 5ms`：0.1169%，其中约 98.9% 是高 wait 比例（>=90%）

### 4.4 纯 CPU 搜索侧 work_time 分布

定义：`work = total - gpu_wait`

- 均值：`0.0998ms`
- 分位数（ms）：
  - p50 `0.0872`
  - p90 `0.1730`
  - p95 `0.2049`
  - p99 `0.2779`
  - p99.9 `0.4134`
  - p99.99 `0.5627`
- 绝大多数集中在 `0.05~0.2ms`（约 94.4%）
- 稀疏长尾存在（最大约 `8.23ms`），但占比极低

### 4.5 work_time 与状态变量关系

- 与 `max_depth` 相关性强（Pearson 约 `0.85`）
- 与 `root_visits_start` 相关性弱到中等（Pearson 约 `0.15`）
- 随搜索树扩大，CPU 时间有增长趋势，但主导因素是路径深度

不同 outcome 的平均 work_time（ms）：

- `new_node_evaluated`：`0.1174`
- `edge_catchup_existing_child`：`0.0749`
- `edge_catchup_new_child`：`0.0958`
- `new_child_race`：`0.0949`

### 4.6 root_turn / 局面形态 bias

不同 `root_turn` 的平均 work_time 差异明显（约 `0.060ms ~ 0.138ms`），说明：

- 树形态对线程工作状态分布有系统影响
- 不能仅用单一全局分布建模全部局面

---

## 5. 当前实现细节与测量可信性

### 5.1 线程侧统计方式

- 每个搜索线程在本地 `rawStatsRows` 里追加行（thread-local vector）
- 每条记录在 playout 结束时写入线程本地内存
- `searchThreadRawStatsMaxRowsPerThread` 控制每线程每次搜索保留上限

### 5.2 写盘时机

- 各线程结束后汇总到 `Search::rawStatsRows`
- 统一在 `Search::maybeFlushRawStatsRowsToFile()` 追加写 TSV
- 写盘发生在搜索流程尾部，而非每次 playout 即时写盘

推论：

- “磁盘写入开销直接污染每条 playout `total_time_ns`”的风险较低
- 每条记录的时间仍可能包含统计代码本身的微小开销（取样不可避免）

---

## 6. trtexec 理论性能摘要（从 JSON 解析）

### 6.1 b18tf

- 可用 streams：1~4，batch：1~32
- 理论 `nnEval/s` 顶部约 `5.1k~5.26k`
- 代表性组合：
  - `s=4,b=4`: 约 `5137 nnEval/s`，平均 latency 约 `3.16ms`
  - `s=2,b=9`: 约 `5258 nnEval/s`，平均 latency 约 `3.48ms`
  - `s=4,b=5`: 约 `5181 nnEval/s`，平均 latency 约 `3.91ms`

结论：

- 不同组合间峰值吞吐差距不大（同一平台约几个百分点）
- 延迟差异会影响达到该吞吐所需的搜索线程数

### 6.2 b28

- 可用 streams：1~4，batch：1~32
- 理论 `nnEval/s` 顶部约 `3.8k~3.87k`
- 最优往往在较大 batch（约 18~22）
- 对应 latency 明显更高（约 11ms~23ms 区间，依 streams 不同）

结论：

- b28 更依赖大 batch 来逼近峰值吞吐
- 大 batch + 高延迟意味着需要更多 search threads 才能喂满

---

## 7. Elo 经验公式与参数选择含义

KataGo benchmark 中使用的经验形式（`PlayUtils::BenchmarkResults::computeEloEffect`）：

- `gain = 250 * log2(visitsPerSecond)`
- `cost = numThreads * 7 * (1600/(800+visitsPerMove))^0.85`
- `elo_effect = gain - cost`

含义：

- 速度提升（visits/s）按对数收益进入 Elo
- 线程数增加有“树质量成本”（经验项）
- 因此不是吞吐最高就一定 Elo 最优，需要平衡线程成本

---

## 8. 已达成的建模共识

- 在当前阶段可近似认为 `visit` 与 `nnEval` 成正比
- `visit` 与 `nnEval` 比例变化若近似常数，则可视为 Elo 上固定平移项
- 选参时主要关注相对比较（不同 `(threads, streams, batch)` 的相对 Elo）

---

## 9. 当前可执行的选参启发式（不依赖重跑）

### 9.1 两阶段筛选

阶段 A（理论面）：

- 从 `trtexec` 选 Pareto 前沿（高 `nnEval/s` + 低 latency）
- 剔除吞吐提升极小但 latency 明显增大的组合

阶段 B（搜索面）：

- 用搜索侧 CPU 分布估算“喂满该组合所需线程数”
- 带入 Elo 经验公式，选 `elo_effect` 最大组合

### 9.2 b18tf 当前建议起点

- 首选：`streams=4, batch=4, threads≈16~18`
- 备选：`streams=2, batch≈8~10, threads≈19~21`
- 不建议直接上很大 batch：吞吐收益有限，线程成本和延迟成本更容易抵消收益

---

## 10. 待补数据与下一步

仍建议在可用 GPU 环境补充：

- 多组 `(threads,streams,batch)` 的 raw stats（至少覆盖候选前沿）
- 同时记录 benchmark 汇总输出（visits/s, nnEvals/s, avgBatchSize, busy-ratio）

之后可做：

- 用分层统计（按 outcome/depth/root_visits 段）建立轻量线程状态模型
- 用同一 Elo 经验目标函数自动输出 Top-K 参数候选
- 仅对 Top-K 做少量实机复验

---

## 11. 附：当前观测对应的 benchmark 运行片段（人工总结）

在 `t16/s4/b4` 的一次运行中，观测到：

- 约 `8474 visits/s`
- 约 `4586 nnEvals/s`
- `avgBatchSize ≈ 2.49`
- 推理线程忙时分布显示高占用接近单/少数流主导

该点用于“理论吞吐 -> 实际搜索吞吐”映射的初始校准，但不是最终定标。

