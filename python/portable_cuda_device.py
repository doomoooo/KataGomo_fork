#!/usr/bin/env python3
"""Small CUDA Runtime device query shared by portable tactic workflows."""

from __future__ import annotations

import os
from typing import Any

from cuda.bindings import runtime


def _unwrap(result: tuple[Any, ...], operation: str) -> Any:
    if not isinstance(result, tuple) or not result:
        raise RuntimeError(f"{operation} returned an unexpected CUDA result")
    error = result[0]
    if error != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed with {getattr(error, 'name', error)}")
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
        return None


def query_cuda_device(device: int) -> dict[str, Any]:
    """Return JSON-safe hardware capabilities for a CUDA runtime ordinal."""
    count = int(_unwrap(runtime.cudaGetDeviceCount(), "cudaGetDeviceCount"))
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
    values = {
        key: value
        for key, attribute in attributes.items()
        if (value := _optional_attribute(attribute, device)) is not None
    }
    direct = {
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
    for key, property_name in direct.items():
        if key not in values and hasattr(properties, property_name):
            values[key] = int(getattr(properties, property_name))
    return {
        "ordinal": device,
        "visibleDeviceCount": count,
        "name": _text(properties.name),
        "computeCapabilityMajor": int(properties.major),
        "computeCapabilityMinor": int(properties.minor),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "totalGlobalMem": int(properties.totalGlobalMem),
        "pci": {
            "domain": int(properties.pciDomainID),
            "bus": int(properties.pciBusID),
            "device": int(properties.pciDeviceID),
        },
        "cudaVisibleDevices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "attributes": values,
        "multiProcessorCount": values["multiProcessorCount"],
        "runtimeVersion": int(_unwrap(
            runtime.cudaRuntimeGetVersion(), "cudaRuntimeGetVersion",
        )),
        "driverVersion": int(_unwrap(
            runtime.cudaDriverGetVersion(), "cudaDriverGetVersion",
        )),
    }
