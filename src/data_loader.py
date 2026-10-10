from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from config.settings import CONFIG, ExperimentConfig


@dataclass
class DatasetSplits:
    """Contenedor estructurado para particiones de datos disjuntas."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


class DataLoader:
    """Carga y división de transacciones financieras."""

    def __init__(self, config: ExperimentConfig = CONFIG) -> None:
        self.config = config

    def load_raw_data(self, file_path: Path | None = None) -> pd.DataFrame:
        """Carga el dataset de transacciones desde disco con validación básica.
        
        Raises:
            FileNotFoundError: Si el archivo creditcard.csv no existe en la ruta.
            ValueError: Si la columna objetivo no está presente.
        """
        path = file_path or self.config.DATA_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el conjunto de datos en: {path}. "
                "Descargue 'creditcard.csv' desde Kaggle y colóquelo en la carpeta 'data/'."
            )

        df = pd.read_csv(path)
        
        if self.config.TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"La columna objetivo '{self.config.TARGET_COLUMN}' no está presente en el archivo."
            )
            
        return df

    def split_data(self, df: pd.DataFrame) -> DatasetSplits:
        """Divide el conjunto de datos en Train, Validation y Test mediante estratificación.
        
        Evita sesgos manteniendo la tasa de desbalance exacta en las tres particiones.
        """
        X = df.drop(columns=[self.config.TARGET_COLUMN])
        y = df[self.config.TARGET_COLUMN].astype(int)

        # 1. Separar conjunto final de prueba (Test)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.config.TEST_SIZE,
            stratify=y,
            random_state=self.config.RANDOM_STATE
        )

        # 2. Separar Entrenamiento (Train) y Validación (Val) a partir de X_temp
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=self.config.VAL_SIZE,
            stratify=y_temp,
            random_state=self.config.RANDOM_STATE
        )

        return DatasetSplits(
            X_train=X_train.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            X_val=X_val.reset_index(drop=True),
            y_val=y_val.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
            y_test=y_test.reset_index(drop=True)
        )