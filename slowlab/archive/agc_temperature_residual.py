"""Frozen linear observation-bias layer for AGC GreenLight air temperature."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Mapping, Sequence

import numpy as np

BASE_FEATURES = (
    "tAir", "tOut", "iGlob", "wind", "uBlScr", "uThScr", "uRoof",
    "qHpsProcessed", "qLedProcessed", "tPipe", "pipeLowActive",
    "tGroPipe", "pipeGrowActive",
)
FEATURE_NAMES = (*BASE_FEATURES, "hour_sin", "hour_cos")


def feature_vector(row: Mapping[str, object], timestamp: datetime) -> np.ndarray:
    values = [float(row[name]) for name in BASE_FEATURES]
    hour = timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600
    angle = 2 * math.pi * hour / 24
    return np.asarray([*values, math.sin(angle), math.cos(angle)], dtype=float)


@dataclass(frozen=True)
class TemperatureResidualModel:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    intercept: float
    coefficients: np.ndarray
    ridge_lambda: float

    def predict_matrix(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("residual feature matrix has wrong shape")
        return self.intercept + ((values - self.mean) / self.scale) @ self.coefficients

    def predict(self, row: Mapping[str, object], timestamp: datetime) -> float:
        return float(self.predict_matrix(feature_vector(row, timestamp)[None, :])[0])

    def to_dict(self) -> dict:
        return {
            "model_type": "standardized_ridge_temperature_observation_residual",
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "intercept": self.intercept,
            "coefficients": self.coefficients.tolist(),
            "ridge_lambda": self.ridge_lambda,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "TemperatureResidualModel":
        return cls(
            tuple(str(x) for x in data["feature_names"]),
            np.asarray(data["mean"], dtype=float),
            np.asarray(data["scale"], dtype=float),
            float(data["intercept"]),
            np.asarray(data["coefficients"], dtype=float),
            float(data["ridge_lambda"]),
        )


def fit_temperature_residual(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    residual: Sequence[float] | np.ndarray,
    *,
    ridge_lambda: float = 10.0,
) -> TemperatureResidualModel:
    x = np.asarray(matrix, dtype=float)
    y = np.asarray(residual, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
        raise ValueError("residual feature matrix has wrong shape")
    if y.shape != (x.shape[0],):
        raise ValueError("residual target has wrong shape")
    if x.shape[0] == 0 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("residual training data must be non-empty and finite")
    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive and finite")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(z)), z))
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(
        design.T @ design + ridge_lambda * penalty,
        design.T @ y,
    )
    return TemperatureResidualModel(
        FEATURE_NAMES, mean, scale, float(beta[0]), beta[1:], ridge_lambda,
    )
