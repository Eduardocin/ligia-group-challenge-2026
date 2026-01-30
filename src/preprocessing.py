"""
Data Preprocessing Pipeline
===========================
Script para limpeza de dados e feature engineering do dataset cardiovascular.
Prepara os dados para a etapa de treinamento de modelos.

Uso:
    python preprocessing.py
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loader import load_data

warnings.filterwarnings("ignore")

# Configuração de paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "Cardiovascular_Disease_Dataset.csv"
INTERIM_DATA_PATH = BASE_DIR / "data" / "interim" / "Cardiovascular_Disease_Dataset_Clean.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza limpeza dos dados:
    - Substitui valores 0 inválidos por NaN
    - Remove linhas com valores faltantes
    - Remove duplicatas
    """
    df_clean = df.copy()

    # Features com valor 0 que deveriam ser NaN
    cols_com_zero_invalido = ["serumcholestrol"]
    df_clean[cols_com_zero_invalido] = df_clean[cols_com_zero_invalido].replace(0, np.nan)

    # Remover linhas com valores faltantes
    df_clean = df_clean.dropna().reset_index(drop=True)

    # Transformar valores de 'slope' de 0 para 1
    df_clean['slope'] = df_clean['slope'].replace(0, 1)

    # Verificar e remover duplicatas
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    return df_clean


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features baseado em dados médicos:
    - age_group: categorização da idade
    - chol_category: categorização do colesterol
    - bp_category: categorização da pressão arterial
    - cholesterol_age_ratio: colesterol ajustado pela idade
    - bp_age_index: índice pressão arterial × idade
    - chronotropic_reserve: reserva cronotrópica
    """
    df_feat = df.copy()

    # Categorização da idade
    df_feat["age_group"] = pd.cut(
        df_feat["age"],
        bins=[0, 40, 60, np.inf],
        labels=[0, 1, 2],  # 0=jovem, 1=meia-idade, 2=idoso
    )
    df_feat["age_group"] = df_feat["age_group"].astype(int)

    # Categorização do colesterol sérico
    df_feat["chol_category"] = pd.cut(
        df_feat["serumcholestrol"],
        bins=[0, 200, 240, np.inf],
        labels=[0, 1, 2],  # 0=normal, 1=limítrofe, 2=alto
    )
    df_feat["chol_category"] = df_feat["chol_category"].astype(int)

    # Categorização da pressão arterial sistólica
    df_feat["bp_category"] = pd.cut(
        df_feat["restingBP"],
        bins=[0, 120, 160, np.inf],
        labels=[0, 1, 2],  # 0=normal, 1=elevada, 2=alta
    )
    df_feat["bp_category"] = df_feat["bp_category"].astype(int)

    # Features derivadas com embasamento médico
    df_feat["cholesterol_age_ratio"] = df_feat["serumcholestrol"] / df_feat["age"]
    df_feat["bp_age_index"] = df_feat["restingBP"] * df_feat["age"]
    df_feat["chronotropic_reserve"] = df_feat["maxheartrate"] / (220 - df_feat["age"])

    return df_feat


def split_train_test(df: pd.DataFrame, test_size: float = 0.3, random_state: int = 42):
    """
    Divide os dados em treino e teste ANTES de normalizar.
    """
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test


def select_features(
    X_train: pd.DataFrame, y_train: pd.Series, correlation_threshold: float = 0.1
) -> tuple[list, dict]:
    """
    Seleciona features baseado em correlação com target.
    Usa APENAS dados de treino para evitar data leakage.
    """
    # Features numéricas
    numerical_features = [
        "age",
        "restingBP",
        "serumcholestrol",
        "maxheartrate",
        "oldpeak",
        "noofmajorvessels",
        "cholesterol_age_ratio",
        "bp_age_index",
        "chronotropic_reserve",
    ]

    # Criar DataFrame temporário para análise
    train_data = X_train.copy()
    train_data["target"] = y_train

    # Calcular correlação
    correlation_matrix = train_data[numerical_features + ["target"]].corr()
    target_corr = correlation_matrix["target"].abs().sort_values(ascending=False)

    # Seleção baseada em threshold
    selected_features_corr = target_corr[target_corr > correlation_threshold].index.tolist()
    selected_features_corr.remove("target") if "target" in selected_features_corr else None

    # Remover features originais se categorizações estiverem presentes
    for feat in ["restingBP", "serumcholestrol", "age"]:
        if feat in selected_features_corr:
            selected_features_corr.remove(feat)

    # Features categóricas
    categorical_features = [
        "gender",
        "chestpain",
        "restingrelectro",
        "slope",
        "age_group",
        "chol_category",
        "bp_category",
    ]

    # Lista final
    selected_features = selected_features_corr + categorical_features

    selected_features_dict = {
        "numerical_selected": selected_features_corr,
        "categorical": categorical_features,
        "all": selected_features,
    }

    return selected_features, selected_features_dict


def normalize_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, list]:
    """
    Normaliza features numéricas usando StandardScaler.
    """
    numerical_features_to_scale = [
        "age",
        "restingBP",
        "serumcholestrol",
        "maxheartrate",
        "oldpeak",
        "noofmajorvessels",
        "cholesterol_age_ratio",
        "bp_age_index",
        "chronotropic_reserve",
    ]

    # Criar e ajustar scaler APENAS no treino
    scaler = StandardScaler()
    scaler.fit(X_train[numerical_features_to_scale])

    # Aplicar em ambos
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_features_to_scale] = scaler.transform(
        X_train[numerical_features_to_scale]
    )
    X_test_scaled[numerical_features_to_scale] = scaler.transform(
        X_test[numerical_features_to_scale]
    )

    return X_train_scaled, X_test_scaled, scaler, numerical_features_to_scale


def save_processed_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    scaler: StandardScaler,
    selected_features: list,
    selected_features_dict: dict,
    output_dir: Path,
) -> dict:
    """
    Salva os dados processados divididos em treino/teste e o scaler.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Salvar conjuntos
    train_file = output_dir / "X_train.csv"
    test_file = output_dir / "X_test.csv"
    y_train_file = output_dir / "y_train.csv"
    y_test_file = output_dir / "y_test.csv"

    # Garantir que todas as colunas sejam numéricas
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_test = X_test.apply(pd.to_numeric, errors="coerce")

    # Preencher possíveis NaN após conversão
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Salvar sem separador customizado (padrão vírgula)
    X_train.to_csv(train_file, index=False)
    X_test.to_csv(test_file, index=False)
    y_train.to_csv(y_train_file, index=False, header=True)
    y_test.to_csv(y_test_file, index=False, header=True)

    # Salvar scaler
    scaler_file = output_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_file)

    # Salvar features
    features_file = output_dir / "selected_features.pkl"
    joblib.dump(selected_features_dict, features_file)

    return {
        "train": train_file,
        "test": test_file,
        "y_train": y_train_file,
        "y_test": y_test_file,
        "scaler": scaler_file,
        "features": features_file,
    }


def run_preprocessing_pipeline() -> dict:
    """
    Executa o pipeline completo de pré-processamento.
    Assume que os dados já foram baixados via data_loader.py
    """
    print("=" * 60)
    print("PIPELINE DE PRÉ-PROCESSAMENTO")
    print("=" * 60)

    # Criar estrutura de pastas se não existir
    (BASE_DIR / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)

    # 1. Carregar dados do CSV (já baixado pelo data_loader.py)
    print("Etapa 1: Carregamento de dados - ", end="")
    if not RAW_DATA_PATH.exists():
        load_data()
    else:
        df_raw = pd.read_csv(RAW_DATA_PATH)
    print("OK")
    print(f"  Dimensões: {df_raw.shape[0]:,} linhas × {df_raw.shape[1]} colunas")

    # 2. Limpeza
    print("Etapa 2: Limpeza de dados - ", end="")
    df_clean = clean_data(df_raw)
    print("OK")

    # 3. Criar features
    print("Etapa 3: Criação de features - ", end="")
    df_features = create_features(df_clean)
    print("OK")
    print(f"Colunas após features: {df_features.columns.tolist()}")

    # 4. Divisão treino/teste
    print("Etapa 4: Divisão Treino/Teste - ", end="")
    X_train, X_test, y_train, y_test = split_train_test(df_features)
    print("OK")
    print(f"  Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")
    print(f"Colunas X_train: {X_train.columns.tolist()}")

    # 5. Seleção de features
    print("Etapa 5: Seleção de Features - ", end="")
    selected_features, selected_features_dict = select_features(X_train, y_train)
    print("OK")
    print(f"  Features selecionadas: {len(selected_features)}")
    print(f"Features selecionadas: {selected_features}")

    # 6. Normalização
    print("Etapa 6: Normalização - ", end="")
    X_train_scaled, X_test_scaled, scaler, scaled_features = normalize_features(X_train, X_test)
    print("OK")
    print(f"Colunas X_train_scaled: {X_train_scaled.columns.tolist()}")

    # 7. Aplicar seleção
    print("Etapa 7: Aplicação Feature Selection - ", end="")
    # Verificar se todas as features selecionadas estão presentes
    missing_features = [feat for feat in selected_features if feat not in X_train_scaled.columns]
    if missing_features:
        print(f"ERRO: Features faltando: {missing_features}")
        # Remover features faltando da lista
        selected_features = [feat for feat in selected_features if feat in X_train_scaled.columns]
        print(f"Features ajustadas: {selected_features}")

    X_train_final = X_train_scaled[selected_features]
    X_test_final = X_test_scaled[selected_features]
    print("OK")

    # 8. Salvar
    print("Etapa 8: Salvamento - ", end="")
    saved_files = save_processed_data(
        X_train_final,
        X_test_final,
        y_train,
        y_test,
        scaler,
        selected_features,
        selected_features_dict,
        PROCESSED_DIR,
    )
    print("OK")

    # Resumo
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    print(f"Total de linhas: {len(df_features)}")
    print(f"Treino: {X_train_final.shape[0]} × {X_train_final.shape[1]} features")
    print(f"Teste: {X_test_final.shape[0]} × {X_test_final.shape[1]} features")
    print("=" * 60)

    return {
        "X_train": X_train_final,
        "X_test": X_test_final,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "selected_features": selected_features_dict,
        "saved_files": saved_files,
    }


if __name__ == "__main__":
    result = run_preprocessing_pipeline()
