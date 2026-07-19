#!/usr/bin/env bash
set -u

STATUS=0

if [ -f pyproject.toml ] || [ -d tests ]; then
  echo "[harness] Running pytest"
  python -m pytest -q || STATUS=$?
fi

if command -v ruff >/dev/null 2>&1; then
  echo "[harness] Running ruff"
  ruff check . || STATUS=$?
fi

if [ -f Cargo.toml ]; then
  echo "[harness] Running cargo test"
  cargo test || STATUS=$?
fi

exit "$STATUS"
