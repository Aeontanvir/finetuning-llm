#!/usr/bin/env bash
# Convenience wrapper: prepare data preview, then train, then test a prompt.
# Usage: bash scripts/run_training.sh [lora|qlora]
set -euo pipefail

MODE="${1:-lora}"
CONFIG="configs/${MODE}.yaml"

echo "==> Previewing dataset formatting"
python src/data_prep.py --config "$CONFIG" --preview

echo "==> Training ($MODE)"
if [ "$MODE" = "qlora" ]; then
  python src/train_qlora.py --config "$CONFIG"
else
  python src/train_lora.py --config "$CONFIG"
fi

echo "==> Done. Try: python src/inference.py --base <base> --adapter results/${MODE}-adapter --prompt 'Hello'"
