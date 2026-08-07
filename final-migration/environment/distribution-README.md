# Prebuilt CUDA-backend distribution

This bundle contains the locally compiled optimizer wheels, their complete
binary Python dependency closure, and the KataGo CUDA-backend executable.
`SHA256SUMS` covers every shipped file.

On a fresh Ubuntu host, from outside the bundle:

```bash
./BUNDLE/installer/deploy-prebuilt.sh ./BUNDLE
```

The installer verifies the complete bundle before changing the host, installs
current CUDA/cuDNN system packages when needed, creates an isolated environment,
and installs only local Python wheels. It does not clone optimizer sources or
compile them. An operational NVIDIA driver is left unchanged; if a new driver
is installed, reboot and rerun the command.

For a host whose recorded CUDA toolkit and cuDNN packages are already present,
`KATAGO_SKIP_SYSTEM_BOOTSTRAP=1` skips all APT operations after verifying those
two packages. This is also useful for a non-destructive bundle validation.

The development setup follows current upstream releases. Deployment instead
keeps the CUDA toolkit major.minor recorded at build time, while allowing
compatible driver and library patch updates.

Use `KATAGO_ENV_ROOT` to place the installed environment in a persistent data
location. Large data must not be placed in a provider's ephemeral system path.
