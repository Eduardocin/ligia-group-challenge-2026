"""
Data Loader Module
==================
Script para carregar e preparar o dataset de doenças cardiovasculares.
Baixa diretamente do Kaggle usando ML Croissant e carrega o arquivo CSV.

Uso:
    from data_loader import load_data
    df = load_data()

    # Ou executar diretamente:
    python data_loader.py
"""

import logging
from pathlib import Path
import warnings

import pandas as pd

# Suprimir warnings gerais
warnings.filterwarnings("ignore")

# Suprimir warnings específicos do ML Croissant (ABSL)
logging.getLogger("absl").setLevel(logging.ERROR)

# Configuração de paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
DATASET_URL = "https://www.kaggle.com/datasets/jocelyndumlao/cardiovascular-disease-dataset/croissant/download"
CSV_FILE = "Cardiovascular_Disease_Dataset.csv"
CSV_PATH = RAW_DATA_DIR / CSV_FILE


def download_with_croissant() -> Path:
    """
    Baixa dataset do Kaggle usando ML Croissant e salva como CSV.

    Returns:
        Path: Caminho do arquivo CSV salvo
    """
    try:
        import mlcroissant as mlc
    except ImportError:
        raise ImportError("mlcroissant não está instalado. Execute: pip install mlcroissant")

    # Criar diretório de destino
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Verificar se já existe
    if CSV_PATH.exists():
        print(f"Dataset já existe em: {CSV_PATH}")
        return CSV_PATH

    # Baixar dataset
    print("Baixando dataset via ML Croissant...")
    croissant_dataset = mlc.Dataset(DATASET_URL)
    record_sets = croissant_dataset.metadata.record_sets

    # Carregar dados
    print("Carregando registros...")
    records = list(croissant_dataset.records(record_set=record_sets[0].uuid))
    df = pd.DataFrame(records)

    # Limpar nomes das colunas
    df.columns = [col.split("/")[-1] if "/" in col else col for col in df.columns]

    # Corrigir coluna patientid (bytes para int)
    if "patientid" in df.columns and len(df) > 0:
        if isinstance(df["patientid"].iloc[0], bytes):
            df["patientid"] = df["patientid"].apply(
                lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
            )
            df["patientid"] = df["patientid"].astype(int)
            print("✓ patientid convertido de bytes para int")

    print(f"✓ Dados carregados: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

    # Salvar CSV
    df.to_csv(CSV_PATH, index=False)
    print(f"✓ Dataset salvo em: {CSV_PATH}")

    return CSV_PATH


def load_data(auto_download: bool = True) -> pd.DataFrame:
    """
    Carrega o dataset de doenças cardiovasculares.

    Args:
        auto_download: Se True, baixa automaticamente se não encontrar

    Returns:
        pd.DataFrame: Dataset carregado
    """
    if not CSV_PATH.exists():
        if auto_download:
            print("Arquivo não encontrado. Iniciando download...")
            try:
                download_with_croissant()
            except Exception as e:
                raise FileNotFoundError(
                    f"Download falhou: {str(e)}\n"
                    f"Baixe manualmente de: https://www.kaggle.com/datasets/jocelyndumlao/cardiovascular-disease-dataset\n"
                    f"E salve em: {CSV_PATH}"
                )
        else:
            raise FileNotFoundError(f"Arquivo não encontrado: {CSV_PATH}")

    # Carregar CSV
    print(f"Carregando dataset de: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    print("✓ Dataset carregado com sucesso!")
    print(f"  Dimensões: {df.shape[0]:,} linhas × {df.shape[1]} colunas")

    return df


def main():
    """Executa o carregamento e exibe informações do dataset."""
    print("=" * 60)
    print("CARREGAMENTO DE DADOS - Cardiovascular Disease Dataset")
    print("=" * 60)

    try:
        load_data(auto_download=True)
        print("=" * 60)
        print("✓ Carregamento concluído!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ ERRO: {str(e)}")
        raise


if __name__ == "__main__":
    main()
