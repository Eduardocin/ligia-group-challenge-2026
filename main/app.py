"""
Sistema de Apoio ao Diagnóstico Cardiovascular
===============================================
Aplicação Streamlit para classificação de doenças cardíacas
utilizando aprendizado de máquina.
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

# Configuração de paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

# Configuração da página
st.set_page_config(
    page_title="Diagnóstico Cardiovascular",
    page_icon="❤️",
    layout="centered"
)


@st.cache_resource
def load_model():
    """Carrega o modelo treinado."""
    model_path = MODELS_DIR / 'best_model.pkl'
    if model_path.exists():
        return joblib.load(model_path)
    return None


@st.cache_resource
def load_scaler():
    """Carrega o scaler treinado."""
    scaler_path = PROCESSED_DIR / 'scaler.pkl'
    if scaler_path.exists():
        return joblib.load(scaler_path)
    return None


def create_derived_features(data: dict) -> dict:
    """Cria features derivadas a partir dos dados de entrada."""
    data['cholesterol_age_ratio'] = data['serumcholestrol'] / data['age']
    data['bp_age_index'] = data['restingBP'] * data['age']
    data['chronotropic_reserve'] = data['maxheartrate'] / (220 - data['age'])
    
    # Categorizações
    if data['age'] <= 40:
        data['age_group'] = 0
    elif data['age'] <= 60:
        data['age_group'] = 1
    else:
        data['age_group'] = 2
    
    if data['serumcholestrol'] <= 200:
        data['chol_category'] = 0
    elif data['serumcholestrol'] <= 240:
        data['chol_category'] = 1
    else:
        data['chol_category'] = 2
    
    if data['restingBP'] <= 120:
        data['bp_category'] = 0
    elif data['restingBP'] <= 160:
        data['bp_category'] = 1
    else:
        data['bp_category'] = 2
    
    return data


def make_prediction(model, scaler, data: dict) -> int:
    """
    Realiza predição usando modelo treinado.
    Retorna 0 (sem doença) ou 1 (com doença).
    """
    features = create_derived_features(data.copy())
    
    # Ordem das features conforme treinamento
    feature_order = [
        'age', 'gender', 'chestpain', 'restingBP', 'serumcholestrol',
        'fastingbloodsugar', 'restingrelectro', 'maxheartrate', 
        'exerciseangia', 'oldpeak', 'slope', 'noofmajorvessels',
        'age_group', 'chol_category', 'bp_category',
        'cholesterol_age_ratio', 'bp_age_index', 'chronotropic_reserve'
    ]
    
    X = pd.DataFrame([features])[feature_order]
    
    # Normalizar features numéricas
    numerical_features = [
        'age', 'restingBP', 'serumcholestrol', 'maxheartrate', 
        'oldpeak', 'noofmajorvessels', 'cholesterol_age_ratio', 
        'bp_age_index', 'chronotropic_reserve'
    ]
    
    if scaler:
        X[numerical_features] = scaler.transform(X[numerical_features])
    
    prediction = model.predict(X)[0]
    
    return int(prediction)


def main():
    """Função principal do aplicativo."""
    
    # CSS Customizado para título
    st.markdown("""
        <style>
        .main-title {
            text-align: center;
            font-size: 2.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ff4b4b 0%, #ff6b6b 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
            padding: 1rem 0;
            font-family: 'Segoe UI', sans-serif;
        }
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 1.1rem;
            margin-bottom: 2rem;
        }
        </style>
        <div class="main-title">
            ❤️ Sistema de Apoio ao Diagnóstico Cardiovascular
        </div>
        <div class="subtitle">
            Análise inteligente baseada em dados clínicos e exames
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Carregar modelo e scaler
    model = load_model()
    scaler = load_scaler()
    
    # Verificar se modelo existe
    if model is None:
        st.error("❌ **Modelo não encontrado!**")
        st.warning(
            f"O modelo treinado não foi encontrado em `{MODELS_DIR / 'best_model.pkl'}`. "
            "Por favor, execute o script de treinamento primeiro."
        )
        st.stop()
    
    if scaler is None:
        st.warning("⚠️ Scaler não encontrado. Usando dados sem normalização.")
    
    # Formulário de entrada
    st.header("📋 Dados do Paciente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Idade", min_value=18, max_value=100, value=50)
        gender = st.selectbox("Sexo", options=[("Masculino", 1), ("Feminino", 0)], format_func=lambda x: x[0])
        restingBP = st.number_input("Pressão Arterial em Repouso (mmHg)", min_value=80, max_value=250, value=120)
        serumcholestrol = st.number_input("Colesterol Sérico (mg/dL)", min_value=100, max_value=600, value=200)
        fastingbloodsugar = st.selectbox(
            "Glicemia em Jejum > 120 mg/dL", 
            options=[("Não", 0), ("Sim", 1)], 
            format_func=lambda x: x[0]
        )
    
    with col2:
        maxheartrate = st.number_input("Frequência Cardíaca Máxima", min_value=60, max_value=220, value=150)
        oldpeak = st.number_input("Depressão ST (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        noofmajorvessels = st.selectbox("Nº de Vasos Principais", options=[0, 1, 2, 3])
        exerciseangia = st.selectbox(
            "Angina Induzida por Exercício", 
            options=[("Não", 0), ("Sim", 1)], 
            format_func=lambda x: x[0]
        )
    
    st.markdown("---")
    st.subheader("📊 Dados do Exame")
    
    col3, col4 = st.columns(2)
    
    with col3:
        chestpain = st.selectbox(
            "Tipo de Dor no Peito",
            options=[
                ("Angina Típica", 0),
                ("Angina Atípica", 1),
                ("Dor Não-Anginosa", 2),
                ("Assintomático", 3)
            ],
            format_func=lambda x: x[0]
        )
        restingrelectro = st.selectbox(
            "ECG em Repouso",
            options=[
                ("Normal", 0),
                ("Anormalidade ST-T", 1),
                ("Hipertrofia Ventricular Esquerda", 2)
            ],
            format_func=lambda x: x[0]
        )
    
    with col4:
        slope = st.selectbox(
            "Inclinação do Segmento ST",
            options=[
                ("Ascendente", 0),
                ("Plano", 1),
                ("Descendente", 2),
                ("Não informado", 3)
            ],
            format_func=lambda x: x[0]
        )
    
    st.markdown("---")
    
    # Botão de predição
    if st.button("🔍 Realizar Análise", type="primary", use_container_width=True):
        
        patient_data = {
            'age': age,
            'gender': gender[1],
            'chestpain': chestpain[1],
            'restingBP': restingBP,
            'serumcholestrol': serumcholestrol,
            'fastingbloodsugar': fastingbloodsugar[1],
            'restingrelectro': restingrelectro[1],
            'maxheartrate': maxheartrate,
            'exerciseangia': exerciseangia[1],
            'oldpeak': oldpeak,
            'slope': slope[1],
            'noofmajorvessels': noofmajorvessels
        }
        
        try:
            prediction = make_prediction(model, scaler, patient_data)
            
            st.markdown("---")
            st.header("📈 Resultado da Análise")
            
            if prediction == 1:
                st.error("⚠️ **DOENÇA CARDIOVASCULAR DETECTADA**")
                st.markdown(
                    "O modelo indica **presença** de doença cardiovascular "
                    "com base nos dados clínicos informados."
                )
            else:
                st.success("✅ **SEM DOENÇA CARDIOVASCULAR**")
                st.markdown(
                    "O modelo indica **ausência** de doença cardiovascular "
                    "com base nos dados clínicos informados."
                )
            
            st.markdown("---")
            st.caption(
                "⚠️ **Aviso:** Este sistema é apenas uma ferramenta de apoio à decisão clínica. "
                "O diagnóstico final deve ser realizado por um profissional de saúde qualificado "
                "com base em avaliação clínica completa."
            )
            
            with st.expander("🔧 Detalhes Técnicos"):
                st.json(patient_data)
                st.write(f"Predição: {prediction}")
        
        except Exception as e:
            st.error(f"❌ **Erro ao realizar predição:** {str(e)}")
            st.error("Verifique se o modelo foi treinado corretamente e se está compatível com as features esperadas.")


if __name__ == "__main__":
    main()