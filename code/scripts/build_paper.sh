#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PAPER="$ROOT/paper"
MODE="${1:-paper}"

case "$MODE" in
  paper)
    SOURCE="main.tex"
    BUILD="$ROOT/output/.build/paper"
    TARGET="$ROOT/output/paper/main.pdf"
    ;;
  ai-details)
    SOURCE="ai_details.tex"
    BUILD="$ROOT/output/.build/ai-details"
    TARGET="$ROOT/output/support/AI工具使用详情.pdf"
    ;;
  *)
    echo "Usage: $0 {paper|ai-details}" >&2
    exit 2
    ;;
esac

mkdir -p "$BUILD" "$(dirname "$TARGET")"
(
  cd "$PAPER"
  latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir="$BUILD" "$SOURCE"
)
cp "$BUILD/${SOURCE%.tex}.pdf" "$TARGET"
echo "Built: $TARGET"
