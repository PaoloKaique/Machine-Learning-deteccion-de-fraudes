from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuración central para el pipeline experimental de detección de fraude."""
    # Rutas
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_PATH: Path = BASE_DIR / "data" / "creditcard.csv"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    
    # Reproducibilidad
    RANDOM_STATE: int = 42
    
    # Partición estratificada (70% train, 15% validation, 15% test)
    TEST_SIZE: float = 0.15
    VAL_SIZE: float = 0.1765  # 0.1765 del 85% restante equivale al ~15% del total
    
    # Definición de variables del dataset ULB
    TARGET_COLUMN: str = "Class"
    CONTINUOUS_FEATURES_TO_SCALE: List[str] = field(
        default_factory=lambda: ["Time", "Amount"]
    )
    PCA_FEATURES: List[str] = field(
        default_factory=lambda: [f"V{i}" for i in range(1, 29)]
    )
    
    # Parámetros de costo asimétrico
    # Un falso negativo (fraude no detectado) cuesta sustancialmente más que una falsa alarma
    COST_FN: float = 10.0
    COST_FP: float = 1.0


CONFIG = ExperimentConfig()