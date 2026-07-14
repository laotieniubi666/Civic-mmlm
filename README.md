# CIVIC-MMLM

CIVIC-MMLM is a PyTorch package for multimodal evidence fusion, selective prediction, constrained decision scoring, and evidence certificates.

## Installation

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## Quick start

```bash
bash scripts/run_demo.sh
```

The demo trains a compact synthetic model, fits a calibration threshold, evaluates it, and writes artifacts to `outputs/demo/`.

## Main commands

```bash
python scripts/train_demo.py --config configs/demo.yaml --output outputs/demo --device cpu
python scripts/calibrate_demo.py --checkpoint outputs/demo/best_model.pt --output outputs/demo/calibrator.json --device cpu
python scripts/evaluate_demo.py --checkpoint outputs/demo/best_model.pt --calibrator outputs/demo/calibrator.json --output outputs/demo --device cpu
python scripts/predict_demo.py --checkpoint outputs/demo/best_model.pt --calibrator outputs/demo/calibrator.json --index 0
```

## Documentation

- `docs/DATASETS.md` — dataset adapter and acquisition notes
- `docs/MODEL_CARD.md` — intended use and safety limits
- `docs/MAPPING_TO_PAPER.md` — component guide

## License

Released under the [MIT License](LICENSE).
