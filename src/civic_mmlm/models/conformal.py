from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from civic_mmlm.utils.io import load_json, save_json


@dataclass
class ConformalState:
    alpha: float
    threshold: float
    score_weights: dict[str, float]
    calibration_size: int


class ConformalAbstentionCalibrator:
    """Split-conformal threshold for the paper's composite nonconformity score."""

    def __init__(
        self,
        alpha: float = 0.2,
        score_weights: dict[str, float] | None = None,
    ) -> None:
        self.alpha = float(alpha)
        self.score_weights = score_weights or {
            "entropy": 1.0,
            "contradiction": 0.7,
            "identification": 0.4,
            "constraint": 1.0,
        }
        self.threshold: float | None = None
        self.calibration_size = 0

    def score(
        self,
        entropy: torch.Tensor,
        contradiction: torch.Tensor,
        identification: torch.Tensor,
        constraint_slack: torch.Tensor,
    ) -> torch.Tensor:
        identification_weakness = (2.0 - identification.to(entropy.dtype)) / 2.0
        return (
            self.score_weights["entropy"] * entropy
            + self.score_weights["contradiction"] * contradiction
            + self.score_weights["identification"] * identification_weakness
            + self.score_weights["constraint"] * constraint_slack
        )

    def fit(self, scores: np.ndarray | torch.Tensor) -> float:
        values = (
            scores.detach().cpu().numpy()
            if isinstance(scores, torch.Tensor)
            else np.asarray(scores)
        )
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size == 0:
            raise ValueError("Cannot fit conformal calibrator on an empty score array")
        n = values.size
        quantile_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.threshold = float(np.quantile(values, quantile_level, method="higher"))
        self.calibration_size = int(n)
        return self.threshold

    def accept(self, scores: torch.Tensor) -> torch.Tensor:
        if self.threshold is None:
            raise RuntimeError("Calibrator must be fitted or loaded before calling accept")
        return scores <= self.threshold

    def save(self, path: str | Path) -> None:
        if self.threshold is None:
            raise RuntimeError("Cannot save an unfitted calibrator")
        state = ConformalState(
            alpha=self.alpha,
            threshold=self.threshold,
            score_weights=self.score_weights,
            calibration_size=self.calibration_size,
        )
        save_json(asdict(state), path)

    @classmethod
    def load(cls, path: str | Path) -> "ConformalAbstentionCalibrator":
        data = load_json(path)
        instance = cls(alpha=data["alpha"], score_weights=data["score_weights"])
        instance.threshold = float(data["threshold"])
        instance.calibration_size = int(data["calibration_size"])
        return instance
