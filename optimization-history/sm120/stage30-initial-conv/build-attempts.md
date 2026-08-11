# Probe build attempts

## Attempt 1

- Result: link failure before any GPU execution.
- Cause: the vendored cuDNN frontend graph header instantiates OSS engine
  helpers that reference NVRTC even though this convolution probe does not
  request the OSS heuristic mode.
- Missing symbols included `nvrtcCreateProgram`, `nvrtcCompileProgram`, and
  `nvrtcGetCUBIN`.
- Resolution: add `-lnvrtc -lcuda` to the standalone probe link. This does not
  alter the graph or candidate set.
