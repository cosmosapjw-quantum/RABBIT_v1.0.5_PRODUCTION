#!/usr/bin/env bash
set -u

# Minimal LaTeX build helper.
# Usage:
#   bash build_latex.sh [main.tex]

MAIN="${1:-}"

if [ -z "$MAIN" ]; then
  if [ -f "main.tex" ]; then
    MAIN="main.tex"
  else
    MAIN="$(find . -maxdepth 3 -name '*.tex' | head -n 1)"
  fi
fi

if [ -z "$MAIN" ]; then
  echo "No .tex file found."
  exit 2
fi

mkdir -p docs/harness/latex_build_logs

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="docs/harness/latex_build_logs/build_${STAMP}.log"

echo "Building: $MAIN"
echo "Log: $LOG"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error "$MAIN" 2>&1 | tee "$LOG"
  STATUS=${PIPESTATUS[0]}
else
  echo "latexmk not found; falling back to pdflatex x2." | tee "$LOG"
  pdflatex -interaction=nonstopmode -halt-on-error "$MAIN" 2>&1 | tee -a "$LOG"
  STATUS1=${PIPESTATUS[0]}
  pdflatex -interaction=nonstopmode -halt-on-error "$MAIN" 2>&1 | tee -a "$LOG"
  STATUS2=${PIPESTATUS[0]}
  if [ "$STATUS1" -eq 0 ] && [ "$STATUS2" -eq 0 ]; then
    STATUS=0
  else
    STATUS=1
  fi
fi

python3 "$(dirname "$0")/check_latex_log.py" "$LOG" || true

exit "$STATUS"
