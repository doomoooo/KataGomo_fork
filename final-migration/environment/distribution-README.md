# Prebuilt CUDA-backend tar

This archive is a non-invasive, relocatable KataGo CUDA-backend distribution.
It contains the CUDA executable, its user-space CUDA/cuDNN/C++/glibc runtime,
the compiled optimizer wheels, their offline Python wheel closure, build/source
manifests, licenses, and `SHA256SUMS`. TensorRT is not included.

The only host dependency is an operational, sufficiently new NVIDIA driver.
The archive deliberately loads `libcuda` from that driver; every other native
runtime library is loaded from the extracted prefix. Nothing is installed into
`/usr`, `/etc`, the package database, or the user's global Python environment.

## Install the runtime

Keep the tar, its `.sha256`, and the adjacent `.install.sh` together, then use
an empty isolated prefix:

```bash
sha256sum --check BUNDLE.tar.install.sh.sha256
./BUNDLE.tar.install.sh BUNDLE.tar /data/katago-runtime
/data/katago-runtime/bin/katago version
```

The installer verifies the tar before extraction and then verifies every file,
the bundled library resolution, driver version, and executable. It refuses
system roots and non-empty prefixes. Removing that one prefix uninstalls the
runtime.

## Optional archived Python tools

The CUDA executable does not need Python. The wheel archive is retained so the
exact locally compiled optimizer inputs can travel with a release. If the
target has the same Python ABI recorded in `metadata/build-platform.txt`, it
can be installed offline into another directory below the isolated prefix:

```bash
/data/katago-runtime/installer/deploy-prebuilt.sh \
  /data/katago-runtime /data/katago-runtime/python-env
```

An ABI mismatch is reported without changing the target; the KataGo executable
remains usable. Development hosts should build a new wheel set with their
current upstream sources instead of silently reusing an incompatible ABI.

The bundle targets Linux x86-64. Its private glibc loader makes it independent
of the target Ubuntu release, subject to the target kernel and NVIDIA driver
requirements recorded in `metadata/`.
