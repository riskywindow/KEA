#!/usr/bin/env bash
# Source this to get a working shell environment for the KEA project.
#   source scripts/env.sh
KEA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export KEA_ROOT
export KEA_PYTHON="$KEA_ROOT/.venv/bin/python"
export LLVM_PREFIX=/usr/local/opt/llvm
export MLIR_DIR="$LLVM_PREFIX/lib/cmake/mlir"
export LLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm"
export PATH="$LLVM_PREFIX/bin:$KEA_ROOT/build/compiler/bin:$KEA_ROOT/build/bin:$PATH"
# The framework Python has no system CA bundle; torchvision weight downloads
# fail with CERTIFICATE_VERIFY_FAILED without this.
if [ -x "$KEA_PYTHON" ]; then
  SSL_CERT_FILE="$("$KEA_PYTHON" -c 'import certifi;print(certifi.where())' 2>/dev/null)"
  [ -n "$SSL_CERT_FILE" ] && export SSL_CERT_FILE REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"
fi
