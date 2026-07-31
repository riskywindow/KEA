#!/usr/bin/env bash
# Fetch a small real image set for KEA calibration and top-1 evaluation.
#
#   scripts/fetch_calibration_data.sh [--force]
#
# Dataset: Imagenette v2 (160px), a 10-class subset of ImageNet-1k published by
# fast.ai.  ~99 MB.  Its 10 WordNet ids map onto real ImageNet-1k class indices
# (see frontend/kea_frontend/data.py), so top-1 against the 1000-way classifier
# is meaningful -- but it is a SUBSET, and every accuracy number the frontend
# reports says so.
#
# Idempotent: if the data is already extracted it does nothing.  If the network
# is unavailable it prints what to do and exits 0 with a clear "degraded"
# message, so it can sit in a build script without breaking it.  Callers that
# need to distinguish should test for the marker file printed at the end.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/models/data"
# Override to use a mirror, or to exercise the offline path in a test.
URL="${KEA_CALIB_URL:-https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz}"
TGZ="$DATA_DIR/imagenette2-160.tgz"
EXTRACTED="$DATA_DIR/imagenette2-160"
MARKER="$DATA_DIR/.imagenette-ok"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

mkdir -p "$DATA_DIR"

# Keep the data out of git without touching the repo-root .gitignore.
if [ ! -f "$DATA_DIR/.gitignore" ]; then
  cat > "$DATA_DIR/.gitignore" <<'EOF'
# Calibration / evaluation data is fetched, never committed.
*
!.gitignore
EOF
fi

if [ "$FORCE" -eq 0 ] && [ -f "$MARKER" ] && [ -d "$EXTRACTED/val" ]; then
  n=$(find "$EXTRACTED/val" -name '*.JPEG' -o -name '*.jpeg' -o -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
  echo "calibration data already present: $EXTRACTED ($n val images)"
  exit 0
fi

echo "fetching Imagenette v2 (160px) -> $DATA_DIR"
echo "  source: $URL"

# certifi CA bundle, same reason as scripts/env.sh
if [ -x "$ROOT/.venv/bin/python" ] && [ -z "${SSL_CERT_FILE:-}" ]; then
  SSL_CERT_FILE="$("$ROOT/.venv/bin/python" -c 'import certifi;print(certifi.where())' 2>/dev/null)"
  [ -n "$SSL_CERT_FILE" ] && export SSL_CERT_FILE CURL_CA_BUNDLE="$SSL_CERT_FILE"
fi

degrade() {
  echo ""
  echo "=============================================================="
  echo " NETWORK UNAVAILABLE -- calibration data was NOT downloaded."
  echo ""
  echo " $1"
  echo ""
  echo " The frontend still works: export scripts fall back to"
  echo " synthetic calibration data and will say so loudly, and will"
  echo " skip accuracy evaluation rather than report a fake number."
  echo ""
  echo " To install the data by hand, drop the tarball at"
  echo "   $TGZ"
  echo " (from $URL)"
  echo " and re-run this script."
  echo "=============================================================="
  exit 0
}

if [ ! -f "$TGZ" ] || [ "$FORCE" -eq 1 ]; then
  if ! command -v curl >/dev/null 2>&1; then
    degrade "curl is not installed."
  fi
  # -f: fail on HTTP error, -L: follow redirects, --retry: transient failures
  if ! curl -fL --retry 3 --retry-delay 2 --connect-timeout 15 \
            -o "$TGZ.part" "$URL"; then
    rm -f "$TGZ.part"
    degrade "curl could not download the archive (offline, DNS, or TLS failure)."
  fi
  mv "$TGZ.part" "$TGZ"
fi

echo "extracting..."
if ! tar -xzf "$TGZ" -C "$DATA_DIR"; then
  echo "ERROR: archive is corrupt; removing it so a re-run re-downloads."
  rm -f "$TGZ"
  exit 1
fi

if [ ! -d "$EXTRACTED/val" ]; then
  echo "ERROR: expected $EXTRACTED/val after extraction; layout changed upstream?"
  exit 1
fi

# The tarball is large and fully reconstructible; drop it once extracted.
rm -f "$TGZ"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$MARKER"

n=$(find "$EXTRACTED/val" \( -name '*.JPEG' -o -name '*.jpeg' -o -name '*.jpg' \) | wc -l | tr -d ' ')
m=$(find "$EXTRACTED/train" \( -name '*.JPEG' -o -name '*.jpeg' -o -name '*.jpg' \) | wc -l | tr -d ' ')
echo "done: $EXTRACTED"
echo "  train: $m images   val: $n images   (10 classes)"
echo "  marker: $MARKER"
