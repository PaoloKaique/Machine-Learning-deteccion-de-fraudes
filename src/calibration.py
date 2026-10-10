from enum import Enum
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


class CalibrationMethod(str, Enum):
    """Métodos de calibración de probabilidades."""
    NONE = "none"
    SIGMOID = "sigmoid"            # Platt Scaling
    ISOTONIC = "isotonic"          # Regresión Isotónica no paramétrica
    ANALYTIC_RUS = "analytic_rus"  # Corrección analítica de Dal Pozzolo et al.


def dal_pozzolo_adjustment(p_sampled: np.ndarray, beta: float) -> np.ndarray:
    """Ajusta analíticamente las probabilidades estimadas bajo submuestreo mayoritario.
    
    Fórmula de Dal Pozzolo et al. (IEEE CIDM 2015):
    P(y=1|x) = (beta * P_s) / (beta * P_s + (1 - P_s))

    Args:
        p_sampled: Probabilidades de fraude estimadas por el modelo entrenado con RUS.
        beta: Tasa de submuestreo de la clase mayoritaria (0 < beta <= 1.0).
    """
    if beta >= 1.0 or beta <= 0.0:
        return p_sampled

    eps = 1e-12
    p_clamped = np.clip(p_sampled, eps, 1.0 - eps)
    numerator = beta * p_clamped
    denominator = numerator + (1.0 - p_clamped)
    return numerator / denominator


class ProbabilityCalibrator:
    """Ajuste y evaluación de calibración sobre modelos probabilísticos."""

    def __init__(self, method: CalibrationMethod = CalibrationMethod.NONE) -> None:
        self.method = method
        self._isotonic_model: IsotonicRegression | None = None
        self._sigmoid_model: LogisticRegression | None = None

    def fit(
        self,
        base_estimator: BaseEstimator,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> "ProbabilityCalibrator":
        """Ajusta el estimador de calibración utilizando el conjunto de validación."""
        if self.method in (CalibrationMethod.NONE, CalibrationMethod.ANALYTIC_RUS):
            return self

        # Obtener probabilidades no calibradas predichas por el estimador base en Validación
        raw_probs = base_estimator.predict_proba(X_val)[:, 1]
        y_arr = np.asarray(y_val)

        if self.method == CalibrationMethod.ISOTONIC:
            self._isotonic_model = IsotonicRegression(
                out_of_bounds="clip",
                y_min=0.0,
                y_max=1.0
            )
            self._isotonic_model.fit(raw_probs, y_arr)

        elif self.method == CalibrationMethod.SIGMOID:
            # Platt Scaling: Regresión logística 1D sobre el logit (log-odds)
            eps = 1e-12
            p_clamped = np.clip(raw_probs, eps, 1.0 - eps)
            logits = np.log(p_clamped / (1.0 - p_clamped)).reshape(-1, 1)

            self._sigmoid_model = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                random_state=42
            )
            self._sigmoid_model.fit(logits, y_arr)

        return self

    def predict_proba(
        self,
        base_estimator: BaseEstimator,
        X: pd.DataFrame,
        beta: float = 1.0
    ) -> np.ndarray:
        """Genera probabilidades calibradas P(Y=1|X) para cualquier conjunto."""
        raw_probs = base_estimator.predict_proba(X)[:, 1]

        if self.method == CalibrationMethod.NONE:
            return raw_probs

        if self.method == CalibrationMethod.ANALYTIC_RUS:
            return dal_pozzolo_adjustment(raw_probs, beta=beta)

        if self.method == CalibrationMethod.ISOTONIC:
            if self._isotonic_model is None:
                raise RuntimeError("El calibrador isotónico no ha sido ajustado con fit().")
            return np.clip(self._isotonic_model.predict(raw_probs), 0.0, 1.0)

        if self.method == CalibrationMethod.SIGMOID:
            if self._sigmoid_model is None:
                raise RuntimeError("El calibrador sigmoideo no ha sido ajustado con fit().")
            eps = 1e-12
            p_clamped = np.clip(raw_probs, eps, 1.0 - eps)
            logits = np.log(p_clamped / (1.0 - p_clamped)).reshape(-1, 1)
            return self._sigmoid_model.predict_proba(logits)[:, 1]

        return raw_probs

    @staticmethod
    def compute_calibration_diagnostics(
        y_true: pd.Series | np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """Calcula métricas diagnósticas y bins de fiabilidad (Reliability Diagram)."""
        y_arr = np.asarray(y_true)
        brier = float(brier_score_loss(y_arr, y_prob))
        loss = float(log_loss(y_arr, y_prob))

        prob_true, prob_pred = calibration_curve(
            y_arr, y_prob, n_bins=n_bins, strategy="uniform"
        )

        bin_counts, _ = np.histogram(y_prob, bins=n_bins, range=(0, 1))
        non_empty = bin_counts > 0
        weights = bin_counts[non_empty] / len(y_prob)

        if len(weights) == len(prob_true):
            ece = float(np.sum(weights * np.abs(prob_true - prob_pred)))
        else:
            ece = float(np.mean(np.abs(prob_true - prob_pred)))

        return {
            "brier_score": brier,
            "log_loss": loss,
            "ece": ece,
            "prob_true": prob_true,
            "prob_pred": prob_pred
        }