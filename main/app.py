import streamlit as st
import pandas as pd
import os

# Título do app
st.title("Análise de Doença Cardiovascular")

# Caminho para o dataset
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'Cardiovascular_Disease_Dataset.csv')

# Carregar os dados
@st.cache_data
def load_data():
    return pd.read_csv(data_path)

df = load_data()

st.header("Prévia dos Dados")
st.write("Primeiras 5 linhas do dataset:")
st.dataframe(df.head())

# Gráfico de linhas (exemplo com índice)
st.header("Gráfico de Linhas (Índice vs Valores)")
if not df.empty:
    st.line_chart(df.select_dtypes(include=['number']).head(50))  # Primeiras 50 linhas