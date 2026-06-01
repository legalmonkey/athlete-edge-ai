from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.explainability import top_raw_factors


def risk_level(probability: float, thresholds: dict[str, float]) -> str:
    if probability >= thresholds.get("high", 0.70):
        return "High Risk"
    if probability >= thresholds.get("low", 0.30):
        return "Moderate Risk"
    return "Low Risk"


def category_confidence(probability: float, thresholds: dict[str, float]) -> float:
    low = thresholds.get("low", 0.30)
    high = thresholds.get("high", 0.70)
    if probability >= high:
        return probability
    if probability < low:
        return 1.0 - probability
    midpoint = (low + high) / 2.0
    half_width = max((high - low) / 2.0, 1e-6)
    return 1.0 - min(abs(probability - midpoint) / half_width, 1.0)


class InjuryRiskPredictor:
    def __init__(self, artifact_path: str | Path | None = None):
        config = load_config()
        path = Path(artifact_path or config.model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found at {path}. Run python -m src.training.train first.")
        self.artifact = joblib.load(path)
        self.model = self.artifact["model"]
        self.feature_columns = self.artifact["feature_columns"]
        self.threshold = float(self.artifact["threshold"])
        self.risk_thresholds = self.artifact["risk_thresholds"]
        self.reference_frame = self.artifact.get("reference_frame")

    def _frame(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        incoming = pd.DataFrame(records)
        base = pd.DataFrame(np.nan, index=incoming.index, columns=self.feature_columns)
        overlap = [col for col in self.feature_columns if col in incoming.columns]
        if overlap:
            base.loc[:, overlap] = incoming[overlap]
        return base

    def predict_batch(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        X = self._frame(records)
        probs = self.model.predict_proba(X)[:, 1]
        results: list[dict[str, Any]] = []
        for idx, prob in enumerate(probs):
            confidence = category_confidence(float(prob), self.risk_thresholds)
            factors = []
            try:
                if self.reference_frame is not None:
                    factors = top_raw_factors(self.model, X.iloc[[idx]], self.reference_frame, self.feature_columns, top_n=3)
            except Exception:
                factors = []
            results.append(
                {
                    "injury_risk_probability": round(float(prob), 4),
                    "risk_level": risk_level(float(prob), self.risk_thresholds),
                    "confidence": round(float(confidence), 4),
                    "decision_threshold": round(self.threshold, 4),
                    "top_risk_factors": factors,
                }
            )
        return results

    def predict_one(self, record: dict[str, Any]) -> dict[str, Any]:
        return self.predict_batch([record])[0]
