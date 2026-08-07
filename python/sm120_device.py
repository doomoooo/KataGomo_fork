#!/usr/bin/env python3
"""CUDA device properties used by the SM120 search and AOT provenance.

The search code has a few compile-time decisions which depend on the amount
of hardware parallelism (most notably CuTe's ``max_active_clusters``).  Keep
those decisions tied to the CUDA runtime device interface instead of to a GPU
marketing name or a fixed SM count.
"""

from __future__ import annotations

import os
from typing import Any

from cuda.bindings import runtime


def _unwrap(result: tuple[Any, ...], operation: str) -> Any:
    if not isinstance(result, tuple) or not result:
        raise RuntimeError(f"{operation} returned an unexpected CUDA result")
    error = result[0]
    if error != runtime.cudaError_t.cudaSuccess:
        error_name = getattr(error, "name", str(error))
        raise RuntimeError(f"{operation} failed with {error_name}")
    if len(result) == 1:
        return None
    return result[1] if len(result) == 2 else result[1:]


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.partition(bytes((0,)))[0].decode("utf-8", errors="replace")
    return str(value)


def _optional_attribute(name: str, device: int) -> int | None:
    attribute = getattr(runtime.cudaDeviceAttr, name, None)
    if attribute is None:
        return None
    try:
        return int(_unwrap(
            runtime.cudaDeviceGetAttribute(attribute, device),
            f"cudaDeviceGetAttribute({name})",
        ))
    except RuntimeError:
        # Some attributes were added after the driver/toolkit combination in
        # use.  The core properties below remain authoritative.
        return None


def query_cuda_device(device: int | None = None) -> dict[str, Any]:
    """Return JSON-safe properties for one CUDA runtime device.

    ``device`` is a CUDA runtime ordinal, including any CUDA_VISIBLE_DEVICES
    remapping.  Querying an explicit ordinal does not change the caller's
    current CUDA device, which keeps AOT generation side-effect free.
    """

    count = int(_unwrap(runtime.cudaGetDeviceCount(), "cudaGetDeviceCount"))
    if count <= 0:
        raise RuntimeError("CUDA runtime reported no devices")
    if device is None:
        device = int(_unwrap(runtime.cudaGetDevice(), "cudaGetDevice"))
    if not 0 <= device < count:
        raise ValueError(f"CUDA device ordinal {device} is outside 0..{count - 1}")

    properties = _unwrap(
        runtime.cudaGetDeviceProperties(device),
        f"cudaGetDeviceProperties({device})",
    )
    attributes = {
        "multiProcessorCount": "cudaDevAttrMultiProcessorCount",
        "sharedMemoryPerMultiprocessor":
            "cudaDevAttrMaxSharedMemoryPerMultiprocessor",
        "regsPerMultiprocessor": "cudaDevAttrMaxRegistersPerMultiprocessor",
        "maxThreadsPerMultiprocessor": "cudaDevAttrMaxThreadsPerMultiProcessor",
        "maxBlocksPerMultiprocessor": "cudaDevAttrMaxBlocksPerMultiprocessor",
        "maxSharedMemoryPerBlockOptin": "cudaDevAttrMaxSharedMemoryPerBlockOptin",
        "l2CacheSize": "cudaDevAttrL2CacheSize",
        "persistingL2CacheMaxSize": "cudaDevAttrMaxPersistingL2CacheSize",
        "accessPolicyMaxWindowSize": "cudaDevAttrMaxAccessPolicyWindowSize",
        "clockRateKHz": "cudaDevAttrClockRate",
        "memoryClockRateKHz": "cudaDevAttrMemoryClockRate",
        "memoryBusWidth": "cudaDevAttrGlobalMemoryBusWidth",
    }
    attribute_values = {
        key: value
        for key, attribute in attributes.items()
        if (value := _optional_attribute(attribute, device)) is not None
    }

    # The direct properties are useful on older CUDA bindings where an
    # attribute enum is not exposed.  The attribute query wins when present.
    property_values = {
        "multiProcessorCount": "multiProcessorCount",
        "sharedMemoryPerMultiprocessor": "sharedMemPerMultiprocessor",
        "regsPerMultiprocessor": "regsPerMultiprocessor",
        "maxThreadsPerMultiprocessor": "maxThreadsPerMultiProcessor",
        "maxBlocksPerMultiprocessor": "maxBlocksPerMultiProcessor",
        "maxSharedMemoryPerBlockOptin": "sharedMemPerBlockOptin",
        "l2CacheSize": "l2CacheSize",
        "persistingL2CacheMaxSize": "persistingL2CacheMaxSize",
        "accessPolicyMaxWindowSize": "accessPolicyMaxWindowSize",
        "memoryBusWidth": "memoryBusWidth",
    }
    for key, property_name in property_values.items():
        if key not in attribute_values and hasattr(properties, property_name):
            attribute_values[key] = int(getattr(properties, property_name))

    result: dict[str, Any] = {
        "ordinal": device,
        "visible_device_count": count,
        "name": _text(properties.name),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_global_memory": int(properties.totalGlobalMem),
        "pci": {
            "domain": int(properties.pciDomainID),
            "bus": int(properties.pciBusID),
            "device": int(properties.pciDeviceID),
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "attributes": attribute_values,
        "runtime_version": int(_unwrap(
            runtime.cudaRuntimeGetVersion(), "cudaRuntimeGetVersion",
        )),
        "driver_version": int(_unwrap(
            runtime.cudaDriverGetVersion(), "cudaDriverGetVersion",
        )),
    }
    # This is the single value consumed by the CuTe scheduler policy.
    result["multiprocessor_count"] = attribute_values["multiProcessorCount"]
    return result


def runtime_max_active_clusters(device: int | None = None) -> int:
    """Return the default one-CTA-per-SM active-cluster count."""

    return int(query_cuda_device(device)["multiprocessor_count"])
