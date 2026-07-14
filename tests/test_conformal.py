from __future__ import annotations

import numpy as np
import torch

from civic_mmlm.models.conformal import ConformalAbstentionCalibrator


def test_conformal_fit_and_accept() -> None:
    calibrator = ConformalAbstentionCalibrator(alpha=0.2)
    threshold = calibrator.fit(np.arange(10, dtype=np.float64))
    assert threshold >= 8.0
    accepted = calibrator.accept(torch.tensor([threshold - 0.1, threshold + 0.1]))
    assert accepted.tolist() == [True, False]
