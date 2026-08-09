"""Choose source-build parallelism without exhausting host memory."""

from __future__ import annotations

import os
import pathlib


_BYTES_PER_HEAVY_JOB = 2 * 1024**3
_BUDGET_NUMERATOR = 3
_BUDGET_DENOMINATOR = 4


def _read_integer(path: pathlib.Path) -> int | None:
    try:
        value = path.read_text().strip()
    except OSError:
        return None
    return int(value) if value.isdigit() else None


def available_memory_bytes() -> int | None:
    """Return the tighter of Linux MemAvailable and cgroup headroom."""
    candidates: list[int] = []
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                candidates.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass

    cgroup_max = _read_integer(pathlib.Path("/sys/fs/cgroup/memory.max"))
    cgroup_current = _read_integer(pathlib.Path("/sys/fs/cgroup/memory.current"))
    if cgroup_max is not None and cgroup_current is not None:
        candidates.append(max(0, cgroup_max - cgroup_current))
    return min(candidates) if candidates else None


def conservative_build_jobs() -> int:
    """Cap nproc by a conservative allowance for heavy C++/CUDA compiles."""
    try:
        cpu_jobs = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_jobs = os.cpu_count() or 1
    available = available_memory_bytes()
    if available is None:
        return max(1, cpu_jobs)
    budget = available * _BUDGET_NUMERATOR // _BUDGET_DENOMINATOR
    memory_jobs = max(1, budget // _BYTES_PER_HEAVY_JOB)
    return max(1, min(cpu_jobs, memory_jobs))
