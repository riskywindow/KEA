# Platform / Toolchain

Verified facts about the development machine. Everything in this repo is built and
tested against exactly this configuration.

## Host

| Item | Value |
| --- | --- |
| OS | macOS 15.6.1 (Darwin 24.6.0) |
| Shell arch | `x86_64` (Rosetta) — `uname -m` reports x86_64 |
| Cores | 12 |
| C++ compiler | Apple clang 16.0.0 (`/usr/bin/clang++`) |
| CMake | 4.2.3 (`/usr/local/bin/cmake`) |
| Ninja | not installed by default |
| Homebrew prefix | `/usr/local` (x86_64 Homebrew) |

## LLVM / MLIR

MLIR **20.1.6** is installed via Homebrew with full development headers and CMake
packages. This is what the out-of-tree `kea` dialect builds against.

```
/usr/local/opt/llvm/bin/mlir-opt
/usr/local/opt/llvm/bin/mlir-tblgen
/usr/local/opt/llvm/bin/mlir-translate
/usr/local/opt/llvm/lib/cmake/mlir/MLIRConfig.cmake
/usr/local/opt/llvm/lib/cmake/llvm/LLVMConfig.cmake
/usr/local/opt/llvm/include/mlir/...
```

Configure any out-of-tree MLIR project with:

```
-DMLIR_DIR=/usr/local/opt/llvm/lib/cmake/mlir
-DLLVM_DIR=/usr/local/opt/llvm/lib/cmake/llvm
```

MLIR 20 is a fast-moving target — in particular the TOSA dialect changed
significantly around this release. Do not trust older documentation; see
`docs/TOSA_NOTES.md`, every statement in which was verified against the
`mlir-opt` binary above.

## Python

The frontend uses a project-local virtualenv at `.venv/`. It is **x86_64**, which
matches the C++ toolchain — do not "fix" this to arm64, the arches must agree for
any future pybind/ctypes interop.

| Package | Version | Note |
| --- | --- | --- |
| Python | 3.12.8 | |
| torch | 2.2.2 | last macOS x86_64 release |
| torchvision | 0.17.2 | supplies MobileNetV2 + pretrained weights |
| onnx | 1.22.0 | |
| numpy | **1.26.4** | pinned `<2`; torch 2.2.2 is built against the NumPy 1.x C API and fails with `_ARRAY_API not found` on NumPy 2 |

Recreate with:

```sh
python3 -m venv .venv
.venv/bin/pip install torch torchvision onnx pillow "numpy<2"
```

Always invoke as `.venv/bin/python`, never a bare `python3` — the system
interpreter has an arch-mismatched torch installed against it.

### TLS certificates

The framework Python ships without a usable CA bundle, so `torchvision` weight
downloads fail with `CERTIFICATE_VERIFY_FAILED`. Point OpenSSL at `certifi`:

```sh
export SSL_CERT_FILE=$(.venv/bin/python -c 'import certifi;print(certifi.where())')
```

`scripts/env.sh` does this for you.

## `scripts/env.sh`

Source it to get everything above at once — venv path, `mlir-opt` and the built
`kea-*` tools on `PATH`, `MLIR_DIR`/`LLVM_DIR` for CMake, and the CA bundle:

```sh
source scripts/env.sh
```

## Cached assets

`~/.cache/torch/hub/checkpoints/mobilenet_v2-b0353104.pth` — pretrained
MobileNetV2 (3,504,872 params), the end-to-end demo network.
