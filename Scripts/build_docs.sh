#!/usr/bin/env bash
# Build a static DocC site for SatinComputeRasteriser into ./docs.
#
# Usage:
#   Scripts/build_docs.sh           # build into ./docs
#   Scripts/build_docs.sh preview   # local preview server
#
# Env overrides:
#   DOCC_TARGET=SatinComputeRasteriser
#   HOSTING_BASE_PATH=Satin-ComputeRasteriser
#   OUTPUT_DIR=docs
#   EMIT_MARKDOWN=1     # per-symbol .md under docs/data/  (requires
#                       # a docc that supports
#                       # --enable-experimental-markdown-output; not in
#                       # Xcode 26's bundled docc)
#   EMIT_LLMS_TXT=1     # write docs/llms.txt by rendering the DocC JSON
#                       # ourselves (see Scripts/docc_json_to_llms.py).
#                       # Works on any docc.
set -euo pipefail

cd "$(dirname "$0")/.."

# Primary target — its hosted site lives at docs/index.html so the
# existing GH Pages URL keeps working.
TARGET="${DOCC_TARGET:-SatinComputeRasteriser}"
# Additional library targets. Each one is hosted under
# docs/<target-lowercased> with its own --hosting-base-path so links
# inside the archive resolve correctly.
EXTRA_TARGETS="${DOCC_EXTRA_TARGETS:-SatinComputeRasteriserStreaming}"
HOSTING_BASE_PATH="${HOSTING_BASE_PATH:-Satin-ComputeRasteriser}"
OUTPUT_DIR="${OUTPUT_DIR:-docs}"

if [[ "${1:-}" == "preview" ]]; then
    exec swift package --disable-sandbox \
        preview-documentation --target "$TARGET"
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

EXTRA_FLAGS=()
if [[ "${EMIT_MARKDOWN:-0}" == "1" ]]; then
    EXTRA_FLAGS+=(--enable-experimental-markdown-output)
fi

swift package --allow-writing-to-directory "$OUTPUT_DIR" \
    generate-documentation --target "$TARGET" \
    --disable-indexing \
    --transform-for-static-hosting \
    --hosting-base-path "$HOSTING_BASE_PATH" \
    --output-path "$OUTPUT_DIR" \
    "${EXTRA_FLAGS[@]}"

if [[ "${EMIT_LLMS_TXT:-0}" == "1" ]]; then
    python3 "$(dirname "$0")/docc_json_to_llms.py" "$OUTPUT_DIR" "$TARGET"
fi

for extra in $EXTRA_TARGETS; do
    sub="$(echo "$extra" | tr '[:upper:]' '[:lower:]')"
    extra_out="$OUTPUT_DIR/$sub"
    mkdir -p "$extra_out"
    swift package --allow-writing-to-directory "$extra_out" \
        generate-documentation --target "$extra" \
        --disable-indexing \
        --transform-for-static-hosting \
        --hosting-base-path "$HOSTING_BASE_PATH/$sub" \
        --output-path "$extra_out" \
        "${EXTRA_FLAGS[@]}"

    if [[ "${EMIT_LLMS_TXT:-0}" == "1" ]]; then
        python3 "$(dirname "$0")/docc_json_to_llms.py" "$extra_out" "$extra"
    fi
done

echo
echo "Docs written to $OUTPUT_DIR/. Open $OUTPUT_DIR/index.html or push to gh-pages."
