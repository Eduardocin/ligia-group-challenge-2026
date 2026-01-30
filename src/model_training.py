"""
Model Training Module
====================
Script para treinar modelo de classificação de doenças cardiovasculares.
Assume que os dados já foram pré-processados via preprocessing.py

Uso:
    python model_training.py
"""

from pathlib import Path
import warnings

import joblib
import pandas as pd
from xgboost import XGBClassifier

from preprocessing import run_preprocessing_pipeline

warnings.filterwarnings("ignore")

# Configuração de paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Arquivos de dados processados
X_TRAIN_PATH = PROCESSED_DIR / "X_train.csv"
Y_TRAIN_PATH = PROCESSED_DIR / "y_train.csv"
X_TEST_PATH = PROCESSED_DIR / "X_test.csv"
Y_TEST_PATH = PROCESSED_DIR / "y_test.csv"

# Modelo e parâmetros
model = XGBClassifier(random_state=42, eval_metric="logloss")

params = {
    "colsample_bytree": 0.7,
    "learning_rate": 0.01,
    "max_depth": 5,
    "n_estimators": 50,
    "subsample": 1.0,
}


def train_model(X_train, y_train, model=model, params=params):
    """Treina o modelo XGBoost com os melhores hiperparâmetros"""
    model.set_params(**params)
    model.fit(X_train, y_train)
    return model


def save_model(model, model_path=None):
    """Salva o modelo treinado em um arquivo .pkl"""
    if model_path is None:
        model_path = MODELS_DIR / "best_model.pkl"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✓ Modelo salvo em: {model_path}")


def run_model_training_pipeline():
    if not PROCESSED_DIR.exists():
        run_preprocessing_pipeline()

    # Carrega os dados
    print("Carregando dados processados...")
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()

    # Treina o modelo
    print("\nTreinando modelo XGBoost...")
    trained_model = train_model(X_train, y_train)
    print("✓ Modelo treinado!")

    # Salva o modelo
    save_model(trained_model)

    print("=" * 60)
    print("✓ TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 60)


if __name__ == "__main__":
    run_model_training_pipeline()
