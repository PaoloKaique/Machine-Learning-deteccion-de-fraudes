import os
import time
from typing import List, Dict, Any
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from config.settings import CONFIG
from src.data_loader import DataLoader
from src.preprocessor import FraudDataPreprocessor
from src.sampling import DataSampler, SamplingStrategy
from src.calibration import ProbabilityCalibrator, CalibrationMethod
from src.evaluation import ModelEvaluator


def build_experiment_pipeline():
    """Define las configuraciones a contrastar en el estudio comparativo."""
    experiments = [
        # --- GAUSSIAN NAIVE BAYES ---
        {
            "name": "GaussianNB (Baseline)",
            "model_type": "nb",
            "model_kwargs": {},
            "sampling": SamplingStrategy.NONE,
            "calibration": CalibrationMethod.NONE,
        },
        {
            "name": "GaussianNB + Isotonic",
            "model_type": "nb",
            "model_kwargs": {},
            "sampling": SamplingStrategy.NONE,
            "calibration": CalibrationMethod.ISOTONIC,
        },
        {
            "name": "GaussianNB + SMOTE",
            "model_type": "nb",
            "model_kwargs": {},
            "sampling": SamplingStrategy.SMOTE,
            "calibration": CalibrationMethod.NONE,
        },
        {
            "name": "GaussianNB + RUS + Dal Pozzolo",
            "model_type": "nb",
            "model_kwargs": {},
            "sampling": SamplingStrategy.UNDERSAMPLING,
            "calibration": CalibrationMethod.ANALYTIC_RUS,
        },

        # --- LOGISTIC REGRESSION ---
        {
            "name": "LogReg (Baseline)",
            "model_type": "lr",
            "model_kwargs": {"max_iter": 1000, "random_state": CONFIG.RANDOM_STATE},
            "sampling": SamplingStrategy.NONE,
            "calibration": CalibrationMethod.NONE,
        },
        {
            "name": "LogReg (Class-Balanced)",
            "model_type": "lr",
            "model_kwargs": {"max_iter": 1000, "class_weight": "balanced", "random_state": CONFIG.RANDOM_STATE},
            "sampling": SamplingStrategy.NONE,
            "calibration": CalibrationMethod.NONE,
        },
        {
            "name": "LogReg + SMOTE",
            "model_type": "lr",
            "model_kwargs": {"max_iter": 1000, "random_state": CONFIG.RANDOM_STATE},
            "sampling": SamplingStrategy.SMOTE,
            "calibration": CalibrationMethod.NONE,
        },
        {
            "name": "LogReg + RUS + Dal Pozzolo",
            "model_type": "lr",
            "model_kwargs": {"max_iter": 1000, "random_state": CONFIG.RANDOM_STATE},
            "sampling": SamplingStrategy.UNDERSAMPLING,
            "calibration": CalibrationMethod.ANALYTIC_RUS,
        },
        {
            "name": "LogReg + RUS + Sigmoid (Platt)",
            "model_type": "lr",
            "model_kwargs": {"max_iter": 1000, "random_state": CONFIG.RANDOM_STATE},
            "sampling": SamplingStrategy.UNDERSAMPLING,
            "calibration": CalibrationMethod.SIGMOID,
        },
    ]
    return experiments


def run():
    print("=" * 80)
    print("INICIANDO PROTOCOLO EXPERIMENTAL: DETECCIÓN DE FRAUDE FINANCIERO")
    print("=" * 80)

    # 1. Carga y Partición Estratificada
    loader = DataLoader(CONFIG)
    raw_df = loader.load_raw_data()
    splits = loader.split_data(raw_df)
    
    # 2. Preprocesamiento (Escalamiento Robusto ajustado solo en Train)
    preprocessor = FraudDataPreprocessor(CONFIG)
    X_train_s, X_val_s, X_test_s = preprocessor.fit_transform_splits(splits)
    
    sampler = DataSampler(CONFIG)
    evaluator = ModelEvaluator(CONFIG)
    experiments = build_experiment_pipeline()
    results_list: List[Dict[str, Any]] = []

    for exp in experiments:
        t0 = time.time()
        print(f"\n[Ejecutando]: {exp['name']}...")

        # A. Remuestreo en Train
        resampled = sampler.apply_sampling(
            X_train_s, splits.y_train, strategy=exp["sampling"], sampling_ratio=0.1
        )

        # B. Instanciar y Entrenar Modelo Base
        if exp["model_type"] == "nb":
            model = GaussianNB(**exp["model_kwargs"])
        else:
            model = LogisticRegression(**exp["model_kwargs"])

        model.fit(resampled.X, resampled.y)

        # C. Ajustar Calibrador (si corresponde) usando Validation
        calibrator = ProbabilityCalibrator(method=exp["calibration"])
        calibrator.fit(model, X_val_s, splits.y_val)

        # D. Predecir Probabilidades en Validación
        val_probs = calibrator.predict_proba(
            model, X_val_s, beta=resampled.majority_sampling_ratio
        )

        # E. Optimización de Umbral sobre Validación (Max F1)
        opt_thresh = evaluator.find_best_threshold_fbeta(
            splits.y_val, val_probs, beta=1.0
        )

        # F. Inferencia Final sobre Test Set (Evaluación ciega)
        test_probs = calibrator.predict_proba(
            model, X_test_s, beta=resampled.majority_sampling_ratio
        )
        metrics = evaluator.evaluate_test_set(
            splits.y_test, test_probs, threshold=opt_thresh.optimal_threshold
        )

        elapsed = time.time() - t0

        results_list.append({
            "Experimento": exp["name"],
            "Estrategia Muestreo": exp["sampling"].value,
            "Calibración": exp["calibration"].value,
            "PR-AUC": round(metrics.pr_auc, 4),
            "ROC-AUC": round(metrics.roc_auc, 4),
            "F1-Score": round(metrics.f1_score, 4),
            "Recall": round(metrics.recall, 4),
            "Precision": round(metrics.precision, 4),
            "Brier Score": round(metrics.brier_score, 6),
            "Log Loss": round(metrics.log_loss, 4),
            "Umbral Opt": round(metrics.threshold, 4),
            "FN": metrics.false_negatives,
            "FP": metrics.false_positives,
            "Costo": round(metrics.total_cost, 2),
            "Tiempo (s)": round(elapsed, 2)
        })

    # Consolidar Resultados
    results_df = pd.DataFrame(results_list)
    
    # Crear directorio outputs si no existe
    CONFIG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = CONFIG.OUTPUT_DIR / "resultados_experimentos.csv"
    results_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("MATRIZ FINAL DE RESULTADOS EXPERIMENTALES (TEST SET)")
    print("=" * 80)
    
    # Visualización limpia de las columnas clave
    cols_display = [
        "Experimento", "PR-AUC", "F1-Score", "Recall", 
        "Precision", "Brier Score", "Umbral Opt", "Costo"
    ]
    print(results_df[cols_display].to_string(index=False))
    print(f"\nResultados guardados exitosamente en: {out_csv}")


if __name__ == "__main__":
    run()