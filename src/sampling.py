from dataclasses import dataclass
from enum import Enum
from typing import Tuple
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from config.settings import CONFIG, ExperimentConfig


class SamplingStrategy(str, Enum):
    """Estrategias de remuestreo disponibles para el entrenamiento."""
    NONE = "none"
    SMOTE = "smote"
    UNDERSAMPLING = "undersampling"


@dataclass
class ResampledData:
    """Estructura que almacena los datos transformados y metadatos del muestreo."""
    X: pd.DataFrame
    y: pd.Series
    strategy: SamplingStrategy
    # Proporción de la clase mayoritaria conservada: beta = N_maj_sampled / N_maj_original
    majority_sampling_ratio: float = 1.0


class DataSampler:
    """Gestor de técnicas de remuestreo para clases desbalanceadas."""

    def __init__(self, config: ExperimentConfig = CONFIG) -> None:
        self.config = config

    def apply_sampling(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        strategy: SamplingStrategy = SamplingStrategy.NONE,
        sampling_ratio: float = 0.1  # Relación clase minoritaria / clase mayoritaria deseada
    ) -> ResampledData:
        """Aplica la técnica de remuestreo seleccionada sobre los datos de entrenamiento.

        Args:
            X_train: Matriz de características de entrenamiento.
            y_train: Vector objetivo binario.
            strategy: Técnica a aplicar ('none', 'smote', 'undersampling').
            sampling_ratio: Proporción deseada post-muestreo (N_min / N_maj).
        """
        if strategy == SamplingStrategy.NONE:
            return ResampledData(
                X=X_train,
                y=y_train,
                strategy=strategy,
                majority_sampling_ratio=1.0
            )

        n_maj_orig = (y_train == 0).sum()

        if strategy == SamplingStrategy.SMOTE:
            # SMOTE genera instancias sintéticas de la clase minoritaria
            smote = SMOTE(
                sampling_strategy=sampling_ratio,
                random_state=self.config.RANDOM_STATE
            )
            X_res, y_res = smote.fit_resample(X_train, y_train)
            return ResampledData(
                X=pd.DataFrame(X_res, columns=X_train.columns),
                y=pd.Series(y_res, name=y_train.name),
                strategy=strategy,
                majority_sampling_ratio=1.0  # La clase mayoritaria no fue reducida
            )

        if strategy == SamplingStrategy.UNDERSAMPLING:
            # Submuestreo aleatorio de transacciones legítimas
            rus = RandomUnderSampler(
                sampling_strategy=sampling_ratio,
                random_state=self.config.RANDOM_STATE
            )
            X_res, y_res = rus.fit_resample(X_train, y_train)
            n_maj_sampled = (y_res == 0).sum()
            beta = float(n_maj_sampled / n_maj_orig)

            return ResampledData(
                X=pd.DataFrame(X_res, columns=X_train.columns),
                y=pd.Series(y_res, name=y_train.name),
                strategy=strategy,
                majority_sampling_ratio=beta
            )

        raise ValueError(f"Estrategia de muestreo no soportada: {strategy}")