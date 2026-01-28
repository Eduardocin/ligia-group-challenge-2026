from xgboost import XGBClassifier
import pandas as pd
import joblib
from pathlib import Path

model = XGBClassifier(random_state=42, eval_metric='logloss')

params = {
    "colsample_bytree": 0.7,
    "learning_rate": 0.01,
    "max_depth": 5,
    "n_estimators": 50,
    "subsample": 1.0
}

def train_model(X_train, y_train, model=model, params=params):
    """Treina o modelo XGBoost com os melhores hiperparâmetros"""
    model.set_params(**params)
    model.fit(X_train, y_train)
    return model

def save_model(model, path='../models/best_model.pkl'):
    """Salva o modelo treinado em um arquivo .pkl"""
    Path(path).parent.mkdir(exist_ok=True)
    joblib.dump(model, path)
    print(f"Modelo salvo em: {path}")

if __name__ == "__main__":
    # Carrega os dados
    X_train = pd.read_csv('../data/processed/X_train.csv', sep=';')
    y_train = pd.read_csv('../data/processed/y_train.csv', sep=';').squeeze()

    # Treina o modelo
    trained_model = train_model(X_train, y_train)

    # Salva o modelo
    save_model(trained_model)
    