#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Micros = std::chrono::duration<double, std::micro>;

enum class Scheme {
  BlockingSync,
  WaiterThread,
  BusyQuery,
  HostCallback,
};

struct Options {
  int device = 0;
  int warmup = 100;
  int iterations = 2000;
  std::vector<int> kernel_us = {0, 10, 100, 1000};
  std::vector<Scheme> schemes = {
      Scheme::BlockingSync,
      Scheme::WaiterThread,
      Scheme::BusyQuery,
      Scheme::HostCallback,
  };
  bool breakdown = false;
  bool sanity_ready = false;
};

struct Stats {
  double avg = 0.0;
  double min = 0.0;
  double p50 = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double max = 0.0;
};

struct SchemeResult {
  std::string name;
  Stats total_us;
  Stats gpu_us;
  Stats tax_us;
};

class EventLatch {
 public:
  auto reset() -> void {
    std::lock_guard lock(mutex_);
    ready_ = false;
  }

  auto signal() -> void {
    {
      std::lock_guard lock(mutex_);
      ready_ = true;
    }
    cv_.notify_one();
  }

  auto wait() -> void {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [&] { return ready_; });
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool ready_ = false;
};

struct IterationTimers {
  Clock::time_point begin;
  Clock::time_point after_record_begin;
  Clock::time_point after_launch;
  Clock::time_point after_record_done;
  Clock::time_point end;
  float gpu_ms = 0.0f;
};

auto scheme_name(Scheme scheme) -> const char* {
  switch (scheme) {
    case Scheme::BlockingSync:
      return "blocking";
    case Scheme::WaiterThread:
      return "waiter";
    case Scheme::BusyQuery:
      return "query";
    case Scheme::HostCallback:
      return "callback";
  }
  return "unknown";
}

auto parse_scheme(std::string_view name) -> Scheme {
  if (name == "blocking") {
    return Scheme::BlockingSync;
  }
  if (name == "waiter") {
    return Scheme::WaiterThread;
  }
  if (name == "query") {
    return Scheme::BusyQuery;
  }
  if (name == "callback") {
    return Scheme::HostCallback;
  }
  std::ostringstream oss;
  oss << "unknown scheme: " << name;
  throw std::runtime_error(oss.str());
}

auto check_cuda(cudaError_t status, const char* what) -> void {
  if (status != cudaSuccess) {
    std::ostringstream oss;
    oss << what << " failed: " << cudaGetErrorName(status) << " (" << cudaGetErrorString(status) << ")";
    throw std::runtime_error(oss.str());
  }
}

__global__ void spin_kernel(std::uint64_t cycles) {
  const std::uint64_t start = clock64();
  while ((clock64() - start) < cycles) {
  }
}

auto parse_kernel_us(const char* value) -> std::vector<int> {
  std::vector<int> values;
  std::stringstream ss(value);
  std::string part;
  while (std::getline(ss, part, ',')) {
    if (part.empty()) {
      continue;
    }
    values.push_back(std::stoi(part));
  }
  if (values.empty()) {
    throw std::runtime_error("empty --kernel-us list");
  }
  return values;
}

auto parse_schemes(const char* value) -> std::vector<Scheme> {
  std::vector<Scheme> values;
  std::stringstream ss(value);
  std::string part;
  while (std::getline(ss, part, ',')) {
    if (part.empty()) {
      continue;
    }
    values.push_back(parse_scheme(part));
  }
  if (values.empty()) {
    throw std::runtime_error("empty --schemes list");
  }
  return values;
}

auto parse_options(int argc, char** argv) -> Options {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--device" && (i + 1) < argc) {
      options.device = std::stoi(argv[++i]);
    } else if (arg == "--warmup" && (i + 1) < argc) {
      options.warmup = std::stoi(argv[++i]);
    } else if (arg == "--iterations" && (i + 1) < argc) {
      options.iterations = std::stoi(argv[++i]);
    } else if (arg == "--kernel-us" && (i + 1) < argc) {
      options.kernel_us = parse_kernel_us(argv[++i]);
    } else if (arg == "--schemes" && (i + 1) < argc) {
      options.schemes = parse_schemes(argv[++i]);
    } else if (arg == "--breakdown") {
      options.breakdown = true;
    } else if (arg == "--sanity-ready") {
      options.sanity_ready = true;
    } else if (arg == "--help") {
      std::cout
          << "Usage: cuda_completion_bench [--device N] [--warmup N] [--iterations N] [--kernel-us a,b,c] [--schemes blocking,waiter,query,callback] [--breakdown] [--sanity-ready]\n";
      std::exit(0);
    } else {
      std::ostringstream oss;
      oss << "unknown argument: " << arg;
      throw std::runtime_error(oss.str());
    }
  }
  return options;
}

auto percentile(const std::vector<double>& sorted, double q) -> double {
  if (sorted.empty()) {
    return 0.0;
  }
  const double pos = q * static_cast<double>(sorted.size() - 1);
  const auto lower = static_cast<std::size_t>(pos);
  const auto upper = std::min(lower + 1, sorted.size() - 1);
  const double weight = pos - static_cast<double>(lower);
  return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

auto make_stats(std::vector<double> values) -> Stats {
  std::sort(values.begin(), values.end());
  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  Stats stats;
  stats.avg = sum / static_cast<double>(values.size());
  stats.min = values.front();
  stats.p50 = percentile(values, 0.50);
  stats.p95 = percentile(values, 0.95);
  stats.p99 = percentile(values, 0.99);
  stats.max = values.back();
  return stats;
}

class WaiterThread {
 public:
  WaiterThread() : thread_([this] { run(); }) {}

  ~WaiterThread() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    cv_.notify_one();
    thread_.join();
  }

  auto wait(cudaEvent_t event) -> void {
    std::unique_lock lock(mutex_);
    pending_event_ = event;
    submit_seq_ += 1;
    const std::uint64_t my_seq = submit_seq_;
    cv_.notify_one();
    cv_.wait(lock, [&] { return done_seq_ >= my_seq; });
  }

 private:
  auto run() -> void {
    std::uint64_t local_seq = 0;
    while (true) {
      cudaEvent_t event = nullptr;
      {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [&] { return stopping_ || submit_seq_ > local_seq; });
        if (stopping_) {
          break;
        }
        event = pending_event_;
        local_seq = submit_seq_;
      }
      check_cuda(cudaEventSynchronize(event), "cudaEventSynchronize");
      {
        std::lock_guard lock(mutex_);
        done_seq_ = local_seq;
      }
      cv_.notify_all();
    }
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread thread_;
  cudaEvent_t pending_event_ = nullptr;
  bool stopping_ = false;
  std::uint64_t submit_seq_ = 0;
  std::uint64_t done_seq_ = 0;
};

struct CallbackState {
  EventLatch* latch = nullptr;
};

void CUDART_CB host_callback(void* user_data) {
  auto* state = static_cast<CallbackState*>(user_data);
  state->latch->signal();
}

struct BenchmarkContext {
  cudaStream_t stream = nullptr;
  cudaEvent_t gpu_begin = nullptr;
  cudaEvent_t gpu_done = nullptr;
  WaiterThread waiter;
  double cycles_per_us = 0.0;

  BenchmarkContext() = default;
  BenchmarkContext(const BenchmarkContext&) = delete;
  auto operator=(const BenchmarkContext&) -> BenchmarkContext& = delete;

  ~BenchmarkContext() {
    if (gpu_done != nullptr) {
      (void)cudaEventDestroy(gpu_done);
    }
    if (gpu_begin != nullptr) {
      (void)cudaEventDestroy(gpu_begin);
    }
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
    }
  }
};

auto launch_and_measure(BenchmarkContext& ctx, Scheme scheme, int kernel_us) -> IterationTimers {
  const std::uint64_t cycles =
      static_cast<std::uint64_t>(std::max(0.0, ctx.cycles_per_us * static_cast<double>(kernel_us)));
  EventLatch callback_latch;
  CallbackState callback_state;
  callback_state.latch = &callback_latch;
  IterationTimers timers;
  timers.begin = Clock::now();

  check_cuda(cudaEventRecord(ctx.gpu_begin, ctx.stream), "cudaEventRecord(gpu_begin)");
  timers.after_record_begin = Clock::now();
  spin_kernel<<<1, 1, 0, ctx.stream>>>(cycles);
  check_cuda(cudaGetLastError(), "spin_kernel");
  timers.after_launch = Clock::now();
  check_cuda(cudaEventRecord(ctx.gpu_done, ctx.stream), "cudaEventRecord(gpu_done)");
  timers.after_record_done = Clock::now();

  switch (scheme) {
    case Scheme::BlockingSync:
      check_cuda(cudaEventSynchronize(ctx.gpu_done), "cudaEventSynchronize");
      break;
    case Scheme::WaiterThread:
      ctx.waiter.wait(ctx.gpu_done);
      break;
    case Scheme::BusyQuery:
      while (true) {
        const cudaError_t status = cudaEventQuery(ctx.gpu_done);
        if (status == cudaSuccess) {
          break;
        }
        if (status != cudaErrorNotReady) {
          check_cuda(status, "cudaEventQuery");
        }
      }
      break;
    case Scheme::HostCallback:
      callback_latch.reset();
      check_cuda(cudaLaunchHostFunc(ctx.stream, host_callback, &callback_state), "cudaLaunchHostFunc");
      callback_latch.wait();
      break;
  }

  const auto end = Clock::now();
  float gpu_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&gpu_ms, ctx.gpu_begin, ctx.gpu_done), "cudaEventElapsedTime");
  timers.end = end;
  timers.gpu_ms = gpu_ms;
  return timers;
}

auto measure_scheme(BenchmarkContext& ctx, Scheme scheme, int warmup, int iterations, int kernel_us, bool breakdown)
    -> SchemeResult {
  for (int i = 0; i < warmup; ++i) {
    (void)launch_and_measure(ctx, scheme, kernel_us);
  }

  std::vector<double> total_us;
  std::vector<double> gpu_us;
  std::vector<double> tax_us;
  std::vector<double> record_begin_us;
  std::vector<double> launch_us;
  std::vector<double> record_done_us;
  std::vector<double> wait_us;
  total_us.reserve(static_cast<std::size_t>(iterations));
  gpu_us.reserve(static_cast<std::size_t>(iterations));
  tax_us.reserve(static_cast<std::size_t>(iterations));
  record_begin_us.reserve(static_cast<std::size_t>(iterations));
  launch_us.reserve(static_cast<std::size_t>(iterations));
  record_done_us.reserve(static_cast<std::size_t>(iterations));
  wait_us.reserve(static_cast<std::size_t>(iterations));

  for (int i = 0; i < iterations; ++i) {
    const IterationTimers timers = launch_and_measure(ctx, scheme, kernel_us);
    const double total = Micros(timers.end - timers.begin).count();
    const double gpu = static_cast<double>(timers.gpu_ms) * 1000.0;
    total_us.push_back(total);
    gpu_us.push_back(gpu);
    tax_us.push_back(total - gpu);
    record_begin_us.push_back(Micros(timers.after_record_begin - timers.begin).count());
    launch_us.push_back(Micros(timers.after_launch - timers.after_record_begin).count());
    record_done_us.push_back(Micros(timers.after_record_done - timers.after_launch).count());
    wait_us.push_back(Micros(timers.end - timers.after_record_done).count());
  }

  std::string name;
  switch (scheme) {
    case Scheme::BlockingSync:
      name = "cudaEventSynchronize(caller)";
      break;
    case Scheme::WaiterThread:
      name = "cudaEventSynchronize(waiter_thread)";
      break;
    case Scheme::BusyQuery:
      name = "cudaEventQuery(busy_poll)";
      break;
    case Scheme::HostCallback:
      name = "cudaLaunchHostFunc";
      break;
  }

  SchemeResult result;
  result.name = std::move(name);
  result.total_us = make_stats(std::move(total_us));
  result.gpu_us = make_stats(std::move(gpu_us));
  result.tax_us = make_stats(std::move(tax_us));
  if (breakdown) {
    std::cout << std::left << std::setw(32) << "record_begin_us"
              << " avg=" << std::setw(10) << std::fixed << std::setprecision(2)
              << make_stats(std::move(record_begin_us)).avg << '\n';
    std::cout << std::left << std::setw(32) << "launch_us"
              << " avg=" << std::setw(10) << make_stats(std::move(launch_us)).avg << '\n';
    std::cout << std::left << std::setw(32) << "record_done_us"
              << " avg=" << std::setw(10) << make_stats(std::move(record_done_us)).avg << '\n';
    std::cout << std::left << std::setw(32) << "wait_us"
              << " avg=" << std::setw(10) << make_stats(std::move(wait_us)).avg << '\n';
  }
  return result;
}

auto print_stats_row(const std::string& label, const Stats& stats) -> void {
  std::cout << std::left << std::setw(32) << label << " avg=" << std::setw(10) << std::fixed << std::setprecision(2)
            << stats.avg << " p50=" << std::setw(10) << stats.p50 << " p95=" << std::setw(10) << stats.p95
            << " p99=" << std::setw(10) << stats.p99 << " max=" << stats.max << '\n';
}

auto measure_ready_event_sync(BenchmarkContext& ctx, int warmup, int iterations) -> Stats {
  for (int i = 0; i < warmup; ++i) {
    check_cuda(cudaEventRecord(ctx.gpu_done, ctx.stream), "cudaEventRecord(ready_sync)");
    check_cuda(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(ready_sync)");
    check_cuda(cudaEventSynchronize(ctx.gpu_done), "cudaEventSynchronize(ready_sync)");
  }

  std::vector<double> values;
  values.reserve(static_cast<std::size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    check_cuda(cudaEventRecord(ctx.gpu_done, ctx.stream), "cudaEventRecord(ready_sync)");
    check_cuda(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(ready_sync)");
    const auto begin = Clock::now();
    check_cuda(cudaEventSynchronize(ctx.gpu_done), "cudaEventSynchronize(ready_sync)");
    const auto end = Clock::now();
    values.push_back(Micros(end - begin).count());
  }
  return make_stats(std::move(values));
}

auto measure_ready_event_query(BenchmarkContext& ctx, int warmup, int iterations) -> Stats {
  for (int i = 0; i < warmup; ++i) {
    check_cuda(cudaEventRecord(ctx.gpu_done, ctx.stream), "cudaEventRecord(ready_query)");
    check_cuda(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(ready_query)");
    check_cuda(cudaEventQuery(ctx.gpu_done), "cudaEventQuery(ready_query)");
  }

  std::vector<double> values;
  values.reserve(static_cast<std::size_t>(iterations));
  for (int i = 0; i < iterations; ++i) {
    check_cuda(cudaEventRecord(ctx.gpu_done, ctx.stream), "cudaEventRecord(ready_query)");
    check_cuda(cudaStreamSynchronize(ctx.stream), "cudaStreamSynchronize(ready_query)");
    const auto begin = Clock::now();
    check_cuda(cudaEventQuery(ctx.gpu_done), "cudaEventQuery(ready_query)");
    const auto end = Clock::now();
    values.push_back(Micros(end - begin).count());
  }
  return make_stats(std::move(values));
}

}  // namespace

auto main(int argc, char** argv) -> int {
  try {
    const Options options = parse_options(argc, argv);
    check_cuda(cudaSetDevice(options.device), "cudaSetDevice");

    cudaDeviceProp prop{};
    check_cuda(cudaGetDeviceProperties(&prop, options.device), "cudaGetDeviceProperties");
    char bus_id[64] = {};
    check_cuda(cudaDeviceGetPCIBusId(bus_id, static_cast<int>(sizeof(bus_id)), options.device), "cudaDeviceGetPCIBusId");

    BenchmarkContext ctx;
    ctx.cycles_per_us = static_cast<double>(prop.clockRate) / 1000.0;
    check_cuda(cudaStreamCreateWithFlags(&ctx.stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    check_cuda(cudaEventCreate(&ctx.gpu_begin), "cudaEventCreate(gpu_begin)");
    check_cuda(cudaEventCreate(&ctx.gpu_done), "cudaEventCreate(gpu_done)");

    std::cout << "device " << options.device << ": " << prop.name << " bus=" << bus_id << '\n';
    std::cout << "warmup=" << options.warmup << " iterations=" << options.iterations << '\n';
    std::cout << "schemes=";
    for (std::size_t i = 0; i < options.schemes.size(); ++i) {
      if (i != 0) {
        std::cout << ',';
      }
      std::cout << scheme_name(options.schemes[i]);
    }
    std::cout << '\n';

    if (options.sanity_ready) {
      std::cout << "\n=== sanity_ready ===\n";
      print_stats_row("cudaEventSynchronize(ready)", measure_ready_event_sync(ctx, options.warmup, options.iterations));
      print_stats_row("cudaEventQuery(ready)", measure_ready_event_query(ctx, options.warmup, options.iterations));
      std::cout << std::flush;
    }

    for (const int kernel_us : options.kernel_us) {
      std::cout << "\n=== kernel_us=" << kernel_us << " ===\n";
      for (const Scheme scheme : options.schemes) {
        const SchemeResult result =
            measure_scheme(ctx, scheme, options.warmup, options.iterations, kernel_us, options.breakdown);
    std::cout << '\n' << result.name << '\n';
        print_stats_row("total_us", result.total_us);
        print_stats_row("gpu_us", result.gpu_us);
        print_stats_row("tax_us", result.tax_us);
        std::cout << std::flush;
      }
    }

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
