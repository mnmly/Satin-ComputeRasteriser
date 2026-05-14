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
#   EMIT_MARKDOWN=1     # per-symbol .md under docs/data/
#   EMIT_LLMS_TXT=1     # also concatenates them into docs/llms.txt
set -euo pipefail

cd "$(dirname "$0")/.."

TARGET="${DOCC_TARGET:-SatinComputeRasteriser}"
HOSTING_BASE_PATH="${HOSTING_BASE_PATH:-Satin-ComputeRasteriser}"
OUTPUT_DIR="${OUTPUT_DIR:-docs}"

if [[ "${1:-}" == "preview" ]]; then
    exec swift package --disable-sandbox \
        preview-documentation --target "$TARGET"
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

EXTRA_FLAGS=()
if [[ "${EMIT_MARKDOWN:-0}" == "1" || "${EMIT_LLMS_TXT:-0}" == "1" ]]; then
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
    LLMS="$OUTPUT_DIR/llms.txt"
    {
        echo "# $TARGET — DocC export for LLM consumption"
        echo
        echo "Generated $(date -u +%FT%TZ) from swift-docc."
        echo
        find "$OUTPUT_DIR/data" -name '*.md' -type f 2>/dev/null \
            | sort \
            | while IFS= read -r f; do
                rel="${f#$OUTPUT_DIR/}"
                echo
                echo "---"
                echo "## $rel"
                echo
                cat "$f"
            done
    } > "$LLMS"
    echo "Wrote $LLMS ($(wc -l < "$LLMS" | tr -d ' ') lines)."
fi

echo
echo "Docs written to $OUTPUT_DIR/. Open $OUTPUT_DIR/index.html or push to gh-pages."
