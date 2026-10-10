from typing import Tuple
import pandas as pd
from sklearn.preprocessing import RobustScaler
from config.settings import CONFIG, ExperimentConfig
from src.data_loader import DatasetSplits


class FraudDataPreprocessor:
    """Preprocesamiento y escalamiento robusto de variables financieras."""

    def __init__(self, config: ExperimentConfig = CONFIG) -> None:
        self.config = config
        self.scaler = RobustScaler()
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame) -> "FraudDataPreprocessor":
        """Calcula mediana e IQR sobre el conjunto de entrenamiento exclusivamente."""
        cols = self.config.CONTINUOUS_FEATURES_TO_SCALE
        self.scaler.fit(X_train[cols])
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica la transformación sobre cualquier partición (Train, Val o Test).
        
        Raises:
            RuntimeError: Si se intenta transformar antes de ejecutar fit().
        """
        if not self.is_fitted:
            raise RuntimeError("El preprocesador debe ejecutarse con .fit() antes de .transform().")

        X_transformed = X.copy()
        cols = self.config.CONTINUOUS_FEATURES_TO_SCALE
        
        # Escalar 'Time' y 'Amount'
        X_transformed[cols] = self.scaler.transform(X[cols])
        return X_transformed

    def fit_transform_splits(
        self, splits: DatasetSplits
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Ajusta sobre X_train y transforma X_train, X_val y X_test de forma coherente."""
        self.fit(splits.X_train)
        
        X_train_scaled = self.transform(splits.X_train)
        X_val_scaled = self.transform(splits.X_val)
        X_test_scaled = self.transform(splits.X_test)

        return X_train_scaled, X_val_scaled, X_test_scaled