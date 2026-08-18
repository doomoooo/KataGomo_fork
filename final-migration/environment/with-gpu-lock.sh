#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: with-gpu-lock.sh [--gpu ORDINAL] [--timeout SECONDS] [--label NAME] -- COMMAND [ARG...]

Serialize GPU work with an advisory flock. All benchmark, replay, NSYS, and
NCU commands sharing a physical GPU must use the same ordinal and lock dir.

Environment:
  KATAGOMO_GPU_LOCK_DIR  Lock directory (default: /tmp/katagomo-gpu-locks)
EOF
}

gpu=0
timeout=""
label=""

while (($# > 0)); do
  case "$1" in
    --gpu)
      (($# >= 2)) || { usage; exit 64; }
      gpu="$2"
      shift 2
      ;;
    --timeout)
      (($# >= 2)) || { usage; exit 64; }
      timeout="$2"
      shift 2
      ;;
    --label)
      (($# >= 2)) || { usage; exit 64; }
      label="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 64
      ;;
  esac
done

[[ "$gpu" =~ ^[0-9]+$ ]] || {
  echo "with-gpu-lock: --gpu must be a nonnegative integer" >&2
  exit 64
}
if [[ -n "$timeout" && ! "$timeout" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "with-gpu-lock: --timeout must be a nonnegative number" >&2
  exit 64
fi
(($# > 0)) || { usage; exit 64; }
command -v flock >/dev/null 2>&1 || {
  echo "with-gpu-lock: flock is not installed" >&2
  exit 69
}

if [[ -z "$label" ]]; then
  label="$(basename -- "$1")"
fi
[[ "$label" != *$'\n'* && "$label" != *$'\r'* ]] || {
  echo "with-gpu-lock: --label may not contain newlines" >&2
  exit 64
}

lock_dir="${KATAGOMO_GPU_LOCK_DIR:-/tmp/katagomo-gpu-locks}"
umask 077
mkdir -p -- "$lock_dir"
lock_path="$lock_dir/gpu-${gpu}.lock"

# Append mode avoids truncating the current owner's metadata before the lock is
# acquired. The owner rewrites metadata only after flock succeeds.
exec 9>>"$lock_path"

if [[ -s "$lock_path" ]]; then
  owner="$(tr '\n' ' ' < "$lock_path")"
  echo "with-gpu-lock: waiting for GPU $gpu ($owner)" >&2
else
  echo "with-gpu-lock: waiting for GPU $gpu" >&2
fi

if [[ -n "$timeout" ]]; then
  if ! flock -x -w "$timeout" 9; then
    echo "with-gpu-lock: timed out waiting for GPU $gpu after ${timeout}s" >&2
    exit 75
  fi
else
  flock -x 9
fi

started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf 'pid=%s label=%s started_utc=%s cwd=%s\n' \
  "$$" "$label" "$started_utc" "$PWD" > "$lock_path"
echo "with-gpu-lock: acquired GPU $gpu for $label (pid $$)" >&2

cleanup() {
  : > "$lock_path"
  echo "with-gpu-lock: released GPU $gpu for $label (pid $$)" >&2
}
trap cleanup EXIT HUP INT TERM

set +e
"$@"
status=$?
set -e
exit "$status"
