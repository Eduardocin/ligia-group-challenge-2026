"""
Data Preprocessing Pipeline
===========================
Script para limpeza de dados e feature engineering do dataset cardiovascular.
Prepara os dados para a etapa de treinamento de modelos.

Uso:
    python data_preprocessing.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

# Configuração de paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / 'data' / 'raw' / 'Cardiovascular_Disease_Dataset.csv'
INTERIM_DATA_PATH = BASE_DIR / 'data' / 'interim' / 'Cardiovascular_Disease_Dataset_Clean.csv'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'


def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Carrega os dados brutos do arquivo CSV."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza limpeza dos dados:
    - Substitui valores 0 inválidos por NaN
    - Remove linhas com valores faltantes
    - Remove duplicatas
    """
    df_clean = df.copy()
    
    # Features com valor 0 que deveriam ser NaN
    cols_com_zero_invalido = ['serumcholestrol']
    df_clean[cols_com_zero_invalido] = df_clean[cols_com_zero_invalido].replace(0, np.nan)
    
    # Remover linhas com valores faltantes
    df_clean = df_clean.dropna().reset_index(drop=True)
    
    # Verificar e remover duplicatas
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    
    return df_clean


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features:
    - age_group: categorização da idade
    - chol_category: categorização do colesterol
    - bp_category: categorização da pressão arterial
    - cholesterol_age_ratio: colesterol ajustado pela idade
    - bp_age_index: índice pressão arterial × idade
    - chronotropic_reserve: reserva cronotrópica
    """
    df_feat = df.copy()
    
    # Categorização da idade
    df_feat['age_group'] = pd.cut(
        df_feat['age'],
        bins=[0, 40, 60, 120],
        labels=[0, 1, 2]  # 0=jovem, 1=meia-idade, 2=idoso
    ).astype(int)
    
    # Categorização do colesterol sérico
    df_feat['chol_category'] = pd.cut(
        df_feat['serumcholestrol'],
        bins=[0, 200, 240, 700],
        labels=[0, 1, 2]  # 0=normal, 1=limítrofe, 2=alto
    ).astype(int)
    
    # Categorização da pressão arterial sistólica
    df_feat['bp_category'] = pd.cut(
        df_feat['restingBP'],
        bins=[0, 120, 160, 240],
        labels=[0, 1, 2]  # 0=normal, 1=elevada, 2=alta
    ).astype(int)
    
    # Features derivadas com embasamento médico
    df_feat['cholesterol_age_ratio'] = df_feat['serumcholestrol'] / df_feat['age']
    df_feat['bp_age_index'] = df_feat['restingBP'] * df_feat['age']
    df_feat['chronotropic_reserve'] = df_feat['maxheartrate'] / (220 - df_feat['age'])
    
    return df_feat


def normalize_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, list]:
    """
    Normaliza features numéricas usando StandardScaler.
    Retorna o DataFrame normalizado, o scaler treinado e a lista de features normalizadas.
    """
    numerical_features_to_scale = [
        'age', 'restingBP', 'serumcholestrol', 'maxheartrate', 
        'oldpeak', 'noofmajorvessels', 'cholesterol_age_ratio', 
        'bp_age_index', 'chronotropic_reserve'
    ]
    
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numerical_features_to_scale] = scaler.fit_transform(df[numerical_features_to_scale])
    
    return df_scaled, scaler, numerical_features_to_scale


def get_selected_features() -> dict:
    """
    Retorna as features selecionadas para modelagem.
    """
    selected_features = {
        'numerical': [
            'maxheartrate', 'oldpeak', 'noofmajorvessels',
            'cholesterol_age_ratio', 'bp_age_index', 'chronotropic_reserve'
        ],
        'categorical': [
            'gender', 'chestpain', 'restingelectro', 'slope',
            'age_group', 'chol_category', 'bp_category'
        ],
        'target': 'target'
    }
    selected_features['all'] = selected_features['numerical'] + selected_features['categorical']
    return selected_features


def save_processed_data(
    df: pd.DataFrame, 
    df_scaled: pd.DataFrame, 
    scaler: StandardScaler,
    output_dir: Path
) -> dict:
    """
    Salva os dados processados e o scaler.
    Retorna um dicionário com os caminhos dos arquivos salvos.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset com features originais + novas
    processed_file = output_dir / 'cardiovascular_processed.csv'
    df.to_csv(processed_file, index=False, sep=";")
    
    # Dataset normalizado
    processed_scaled_file = output_dir / 'cardiovascular_processed_scaled.csv'
    df_scaled.to_csv(processed_scaled_file, index=False, sep=";")
    
    # Salvar scaler para uso em produção/inferência
    scaler_file = output_dir / 'scaler.pkl'
    joblib.dump(scaler, scaler_file)
    
    # Salvar metadados das features selecionadas
    selected_features = get_selected_features()
    features_file = output_dir / 'selected_features.pkl'
    joblib.dump(selected_features, features_file)
    
    return {
        'processed': processed_file,
        'scaled': processed_scaled_file,
        'scaler': scaler_file,
        'features': features_file
    }


def run_pipeline() -> dict:
    """
    Executa o pipeline completo de pré-processamento.
    Retorna um dicionário com os resultados e caminhos dos arquivos.
    """
    print("=" * 60)
    print("PIPELINE DE PRÉ-PROCESSAMENTO")
    print("=" * 60)
    
    # 1. Carregar dados brutos
    print("Etapa 1: Carregamento de dados - ", end="")
    df_raw = load_raw_data(RAW_DATA_PATH)
    print("OK")
    
    # 2. Limpeza dos dados
    print("Etapa 2: Limpeza de dados - ", end="")
    df_clean = clean_data(df_raw)
    print("OK")
    
    # 3. Feature Engineering
    print("Etapa 3: Feature Engineering - ", end="")
    df_features = create_features(df_clean)
    print("OK")
    
    # 4. Normalização
    print("Etapa 4: Normalização - ", end="")
    df_scaled, scaler, scaled_features = normalize_features(df_features)
    print("OK")
    
    # 5. Salvar dados processados
    print("Etapa 5: Salvamento - ", end="")
    saved_files = save_processed_data(df_features, df_scaled, scaler, PROCESSED_DIR)
    print("OK")
    
    # Resumo final
    selected_features = get_selected_features()
    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    print(f"Linhas processadas: {len(df_features)}")
    print(f"Total de features: {df_features.shape[1]}")
    print(f"Features selecionadas: {len(selected_features['all'])}")
    print("=" * 60)
    
    return {
        'df': df_features,
        'df_scaled': df_scaled,
        'scaler': scaler,
        'selected_features': selected_features,
        'saved_files': saved_files
    }


if __name__ == "__main__":
    result = run_pipeline()