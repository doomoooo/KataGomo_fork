#!/usr/bin/env python3

import os
from pathlib import Path
import shlex
import subprocess
from typing import Dict, Optional, Sequence, Set


DEFAULT_ENV_KEYS = (
    "TENSORRT_ROOT",
    "KATAGO_BIN_PATH",
    "KATAGO_MODEL_PATH",
    "KATAGO_CONFIG_PATH",
    "TRT_DEVICE_ID",
    "KATAGO_GLOBAL_PERF_PROFILE",
    "KATAGO_MONITOR_SOCKET_PATH",
    "KATAGO_MONITOR_HTTP_HOST",
    "KATAGO_MONITOR_HTTP_PORT",
    "KATAGO_MONITOR_INTERVAL_MS",
)

DEFAULT_PATH_KEYS = {
    "TENSORRT_ROOT",
    "KATAGO_BIN_PATH",
    "KATAGO_MODEL_PATH",
    "KATAGO_CONFIG_PATH",
    "KATAGO_MONITOR_SOCKET_PATH",
}


def resolve_path_with_base(raw: str, base_dir: Path) -> str:
    expanded = os.path.expanduser(os.path.expandvars(raw.strip()))
    path = Path(expanded)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def load_env_sh_defaults(
    env_sh_path: Path,
    keys: Optional[Sequence[str]] = None,
    path_keys: Optional[Set[str]] = None,
) -> Dict[str, str]:
    if not env_sh_path.exists():
        return {}

    actual_keys = list(keys or DEFAULT_ENV_KEYS)
    actual_path_keys = set(path_keys or DEFAULT_PATH_KEYS)
    var_expr = " ".join(f'"${{{key}-}}"' for key in actual_keys)
    shell_cmd = f"source {shlex.quote(str(env_sh_path))} >/dev/null 2>&1 && printf '%s\\n' {var_expr}"
    proc = subprocess.run(
        ["bash", "-lc", shell_cmd],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return {}

    lines = (proc.stdout or "").splitlines()
    out: Dict[str, str] = {}
    base_dir = env_sh_path.parent
    for idx, key in enumerate(actual_keys):
        if idx >= len(lines):
            break
        value = lines[idx].strip()
        if not value:
            continue
        if key in actual_path_keys:
            out[key] = resolve_path_with_base(value, base_dir)
        else:
            out[key] = value
    return out
