# Detección de Fraudes con Tarjetas de Crédito

Proyecto del curso de Modelos Probabilísticos (UNSAAC) para detectar
transacciones fraudulentas usando técnicas de Machine Learning.

## Estructura del proyecto
├── config/ # Configuración y parámetros (settings.py)
├── data/ # Datos (no incluidos en el repo)
├── notebooks/ # Exploración y análisis
├── outputs/ # Resultados de los experimentos
├── src/
│ ├── data_loader.py # Carga de datos
│ ├── preprocessor.py # Preprocesamiento
│ ├── sampling.py # Técnicas de muestreo
│ ├── calibration.py # Calibración de probabilidades
│ └── evaluation.py # Métricas de evaluación
├── run_experiments.py # Script principal
└── requirements.txt

Descarga `creditcard.csv` desde Kaggle (*Credit Card Fraud Detection*)
y colócalo en la carpeta `data/`.

## Instalación

```bash
git clone https://github.com/PaoloKaique/Machine-Learning-deteccion-de-fraudes.git
cd Machine-Learning-deteccion-de-fraudes
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
python run_experiments.py
```

Los resultados se guardan en `outputs/`.

## Autor

Paolo Kaique, UNSAAC
