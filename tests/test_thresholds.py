import numpy as np

from src.evaluation.metrics import optimize_threshold
from src.inference import risk_level


def test_risk_level_mapping():
    thresholds = {"low": 0.3, "high": 0.7}
    assert risk_level(0.1, thresholds) == "Low Risk"
    assert risk_level(0.5, thresholds) == "Moderate Risk"
    assert risk_level(0.9, thresholds) == "High Risk"


def test_threshold_optimizer_prefers_precision():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    threshold, metrics = optimize_threshold(y, p, min_precision=0.8, max_fpr=0.25, grid_size=30)
    assert 0.0 < threshold < 1.0
    assert metrics["precision"] >= 0.8
