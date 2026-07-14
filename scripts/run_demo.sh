#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
python "$ROOT/scripts/train_demo.py" --config "$ROOT/configs/demo.yaml" --output "$ROOT/outputs/demo" --device cpu
python "$ROOT/scripts/calibrate_demo.py" --checkpoint "$ROOT/outputs/demo/best_model.pt" --output "$ROOT/outputs/demo/calibrator.json" --device cpu
python "$ROOT/scripts/evaluate_demo.py" --checkpoint "$ROOT/outputs/demo/best_model.pt" --calibrator "$ROOT/outputs/demo/calibrator.json" --output "$ROOT/outputs/demo" --device cpu
python "$ROOT/scripts/predict_demo.py" --checkpoint "$ROOT/outputs/demo/best_model.pt" --calibrator "$ROOT/outputs/demo/calibrator.json" --index 0
