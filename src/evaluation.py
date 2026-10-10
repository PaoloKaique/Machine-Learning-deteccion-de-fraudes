from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    fbeta_score,
)
from config.settings import CONFIG, ExperimentConfig


@dataclass(frozen=True)
class ThresholdOptimizationResult:
    """Resultado del ajuste de umbral en el conjunto de validación."""
    optimal_threshold: float
    criterion: str
    val_score: float


@dataclass(frozen=True)
class EvaluationMetrics:
    """Métricas consolidadas de desempeño en prueba."""
    # Métricas Probabilísticas (independientes del umbral)
    pr_auc: float
    roc_auc: float
    brier_score: float
    log_loss: float
    
    # Métricas de Decisión (en el umbral óptimo p*)
    threshold: float
    precision: float
    recall: float
    f1_score: float
    f2_score: float
    
    # Matriz de Confusión y Costo
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_cost: float


class ModelEvaluator:
    """Evaluación rigurosa de modelos probabilísticos y optimización de umbrales."""

    def __init__(self, config: ExperimentConfig = CONFIG) -> None:
        self.config = config

    def find_best_threshold_fbeta(
        self,
        y_val: pd.Series | np.ndarray,
        val_probs: np.ndarray,
        beta: float = 1.0
    ) -> ThresholdOptimizationResult:
        """Encuentra el umbral que maximiza F-beta en validación."""
        precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
        
        # Evitar división por cero en el cálculo vectorial de F-beta
        numerator = (1 + beta**2) * (precisions * recalls)
        denominator = (beta**2 * precisions) + recalls
        
        with np.errstate(divide="ignore", invalid="ignore"):
            f_scores = np.where(denominator > 0, numerator / denominator, 0.0)

        # precision_recall_curve devuelve thresholds de longitud len(precisions) - 1
        f_scores_for_thresh = f_scores[:-1]
        best_idx = int(np.argmax(f_scores_for_thresh))
        best_threshold = float(thresholds[best_idx])
        best_score = float(f_scores_for_thresh[best_idx])

        return ThresholdOptimizationResult(
            optimal_threshold=best_threshold,
            criterion=f"Max F{beta:.1f}",
            val_score=best_score
        )

    def find_best_threshold_cost(
        self,
        y_val: pd.Series | np.ndarray,
        val_probs: np.ndarray,
        cost_fn: float | None = None,
        cost_fp: float | None = None,
        num_thresholds: int = 200
    ) -> ThresholdOptimizationResult:
        """Encuentra el umbral que minimiza la pérdida económica en validación.
        
        Costo = (C_FN * FN) + (C_FP * FP)
        """
        c_fn = cost_fn if cost_fn is not None else self.config.COST_FN
        c_fp = cost_fp if cost_fp is not None else self.config.COST_FP
        
        y_arr = np.asarray(y_val)
        candidate_thresholds = np.linspace(0.001, 0.999, num_thresholds)
        min_cost = float("inf")
        best_thresh = 0.5

        for thresh in candidate_thresholds:
            preds = (val_probs >= thresh).astype(int)
            fn = np.sum((y_arr == 1) & (preds == 0))
            fp = np.sum((y_arr == 0) & (preds == 1))
            total_cost = (c_fn * fn) + (c_fp * fp)
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_thresh = float(thresh)

        return ThresholdOptimizationResult(
            optimal_threshold=best_thresh,
            criterion="Min Cost",
            val_score=min_cost
        )

    def evaluate_test_set(
        self,
        y_test: pd.Series | np.ndarray,
        test_probs: np.ndarray,
        threshold: float
    ) -> EvaluationMetrics:
        """Calcula el vector exhaustivo de métricas de desempeño sobre Test."""
        y_arr = np.asarray(y_test)
        
        # 1. Métricas Probabilísticas continuas
        pr_auc = float(average_precision_score(y_arr, test_probs))
        roc_auc = float(roc_auc_score(y_arr, test_probs))
        brier = float(brier_score_loss(y_arr, test_probs))
        loss = float(log_loss(y_arr, test_probs))

        # 2. Clasificación binaria en el umbral seleccionado
        y_pred = (test_probs >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel()
        
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(fbeta_score(y_arr, y_pred, beta=1.0, zero_division=0))
        f2 = float(fbeta_score(y_arr, y_pred, beta=2.0, zero_division=0))

        cost = float((self.config.COST_FN * fn) + (self.config.COST_FP * fp))

        return EvaluationMetrics(
            pr_auc=pr_auc,
            roc_auc=roc_auc,
            brier_score=brier,
            log_loss=loss,
            threshold=float(threshold),
            precision=precision,
            recall=recall,
            f1_score=f1,
            f2_score=f2,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            total_cost=cost
        )