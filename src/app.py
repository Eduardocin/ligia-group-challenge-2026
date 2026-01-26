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
import pdfplumber
import re

# Configurações iniciais
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

st.set_page_config(page_title="Diagnóstico Cardiovascular", page_icon="❤️", layout="centered")

# Dicionários de conversão de dados
MAP_GENDER = {"Masculino": 1, "Feminino": 0}
MAP_ANGINA = {"Sim": 1, "Não": 0}
MAP_CHEST_PAIN = {
    "Angina Típica": 0,
    "Angina Atípica": 1,
    "Dor Não-Anginosa": 2,
    "Assintomático": 3
}
MAP_ECG = {
    "Normal": 0,
    "Anormalidade ST-T": 1,
    "Hipertrofia Ventricular": 2
}
MAP_SLOPE = {
    "Ascendente": 1,
    "Plano": 2,
    "Descendente": 3,
    "Não informado": None
}

# Controle de estado da interface
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "📝 Inserção Manual"

# Carregamento do modelo e do scaler
@st.cache_resource
def load_model():
    model_path = MODELS_DIR / 'best_model.pkl'
    if model_path.exists(): return joblib.load(model_path)
    return None

@st.cache_resource
def load_scaler():
    scaler_path = PROCESSED_DIR / 'scaler.pkl'
    if scaler_path.exists(): return joblib.load(scaler_path)
    return None

@st.cache_resource
def load_selected_features():
    features_path = PROCESSED_DIR / 'selected_features.pkl'
    if features_path.exists(): return joblib.load(features_path)
    return None

# Processamento e predição
def make_prediction(model, scaler, data: dict) -> int:
    safe_data = data.copy()
    defaults = {'age': 50, 'serumcholestrol': 200, 'restingBP': 120, 'maxheartrate': 150}
    
    for k, v in safe_data.items():
        if v is None: safe_data[k] = defaults.get(k, 0)

    safe_data['cholesterol_age_ratio'] = safe_data['serumcholestrol'] / safe_data['age']
    safe_data['bp_age_index'] = safe_data['restingBP'] * safe_data['age']
    safe_data['chronotropic_reserve'] = safe_data['maxheartrate'] / (220 - safe_data['age'])
    
    safe_data['age_group'] = 0 if safe_data['age'] <= 40 else 1 if safe_data['age'] <= 60 else 2
    safe_data['chol_category'] = 0 if safe_data['serumcholestrol'] <= 200 else 1 if safe_data['serumcholestrol'] <= 240 else 2
    safe_data['bp_category'] = 0 if safe_data['restingBP'] <= 120 else 1 if safe_data['restingBP'] <= 160 else 2
    
    selected_features = load_selected_features()
    if not selected_features:
        raise ValueError("Selected features not found")
    
    X = pd.DataFrame([safe_data])
    
    numerical_features = ['age', 'restingBP', 'serumcholestrol', 
                          'maxheartrate', 'oldpeak', 'noofmajorvessels', 
                          'cholesterol_age_ratio', 'bp_age_index', 'chronotropic_reserve']
    
    if scaler: 
        X[numerical_features] = scaler.transform(X[numerical_features])

    X = X[selected_features]
    
    return int(model.predict(X)[0])

# Leitura e extração do PDF
def extract_features_from_pdfs(uploaded_files):
    data = {}
    full_text = ""
    for f in uploaded_files:
        with pdfplumber.open(f) as pdf:
            for page in pdf.pages:
                txt = page.extract_text(layout=True)
                if txt: full_text += txt + "\n"

    match = re.search(r"\((\d+)\s*anos\)|Idade:\s*(\d+)\s*anos", full_text)
    data['age'] = int(match.group(1) or match.group(2)) if match else None

    if re.search(r"\[X\]\s*M", full_text): data['gender'] = "Masculino"
    elif re.search(r"\[X\]\s*F", full_text): data['gender'] = "Feminino"
    else: data['gender'] = None

    match = re.search(r"PA:\s*(\d+)\s*/\s*(\d+)", full_text)
    data['restingBP'] = int(match.group(1)) if match else None

    match = re.search(r"COLESTEROL TOTAL\s+(\d+)", full_text)
    data['serumcholestrol'] = int(match.group(1)) if match else None

    match = re.search(r"GLICOSE \(JEJUM\)\s+(\d+)", full_text)
    data['fastingbloodsugar'] = int(match.group(1)) if match else None

    if re.search(r"\[X\]\s*Angina Típica", full_text, re.IGNORECASE): data['chestpain'] = "Angina Típica"
    elif re.search(r"\[X\]\s*Angina Atípica", full_text, re.IGNORECASE): data['chestpain'] = "Angina Atípica"
    elif re.search(r"\[X\]\s*Dor Não-Anginosa", full_text, re.IGNORECASE): data['chestpain'] = "Dor Não-Anginosa"
    elif re.search(r"\[X\]\s*Assintomático", full_text, re.IGNORECASE): data['chestpain'] = "Assintomático"
    else: data['chestpain'] = None

    if re.search(r"Achados do.*?\[X\]\s*Normal", full_text, re.DOTALL): data['restingrelectro'] = "Normal"
    elif re.search(r"\[X\]\s*Anormalidade de Onda ST-T", full_text): data['restingrelectro'] = "Anormalidade ST-T"
    elif re.search(r"\[X\]\s*Hipertrofia Ventricular", full_text): data['restingrelectro'] = "Hipertrofia Ventricular"
    else: data['restingrelectro'] = None

    match = re.search(r"Peak HR\):\s*(\d+)", full_text)
    data['maxheartrate'] = int(match.group(1)) if match else None

    if re.search(r"Angina durante esforço:.*?\[X]\s*Sim", full_text, re.DOTALL): data['exerciseangia'] = "Sim"
    elif re.search(r"Angina durante esforço:.*?\[X]\s*Não", full_text, re.DOTALL): data['exerciseangia'] = "Não"
    else: data['exerciseangia'] = None

    match = re.search(r"Depressão de ST:\s*(\d+[.,]?\d*)", full_text)
    data['oldpeak'] = float(match.group(1).replace(',', '.')) if match else None

    if re.search(r"Morfologia.*?\[X\]\s*Ascendente", full_text, re.DOTALL): data['slope'] = "Ascendente"
    elif re.search(r"Morfologia.*?\[X\]\s*Horizontal", full_text, re.DOTALL): data['slope'] = "Plano"
    elif re.search(r"Morfologia.*?\[X\]\s*Descendente", full_text, re.DOTALL): data['slope'] = "Descendente"
    else: data['slope'] = None

    match = re.search(r"CONCLUSÃO DO EXAME:.*?Presença de\s*(\d+)", full_text, re.DOTALL)
    data['noofmajorvessels'] = int(match.group(1)) if match else None

    return data

# Funções de suporte ao formulário
def initialize_form_state(data_source, prefix):
    fields = [
        'age', 'gender', 'restingBP', 'serumcholestrol', 'fastingbloodsugar',
        'maxheartrate', 'oldpeak', 'noofmajorvessels', 'exerciseangia',
        'chestpain', 'restingrelectro', 'slope'
    ]
    for field in fields:
        key = f"{prefix}_{field}"
        if key not in st.session_state:
            st.session_state[key] = data_source.get(field, None)

def get_label(text, key):
    val = st.session_state.get(key)
    if val is None:
        return f":red[{text}]"
    return text

# Interface do formulário de entrada
def render_form(prefix):
    col1, col2 = st.columns(2)
    k = lambda x: f"{prefix}_{x}"

    with col1:
        st.number_input(get_label("Idade", k('age')), min_value=18, max_value=100, 
                        key=k('age'), value=st.session_state[k('age')], placeholder="Digite...")
        
        opts = list(MAP_GENDER.keys())
        curr = st.session_state.get(k('gender'))
        idx = opts.index(curr) if curr in opts else None
        st.selectbox(get_label("Sexo", k('gender')), opts, index=idx, key=k('gender'), placeholder="Selecione...")

        st.number_input(get_label("Pressão Arterial", k('restingBP')), 80, 250, 
                        key=k('restingBP'), value=st.session_state[k('restingBP')], placeholder="Ex: 120")
        
        st.number_input(get_label("Colesterol (mg/dL)", k('serumcholestrol')), 100, 600, 
                        key=k('serumcholestrol'), value=st.session_state[k('serumcholestrol')], placeholder="Ex: 200")
        
        st.number_input(get_label("Glicemia Jejum (mg/dL)", k('fastingbloodsugar')), 0, 500, 
                        key=k('fastingbloodsugar'), value=st.session_state[k('fastingbloodsugar')], placeholder="Ex: 90")

    with col2:
        st.number_input(get_label("Frequência Cardíaca Máx", k('maxheartrate')), 60, 220, 
                        key=k('maxheartrate'), value=st.session_state[k('maxheartrate')], placeholder="Ex: 150")
        
        st.number_input(get_label("Depressão ST (oldpeak)", k('oldpeak')), 0.0, 10.0, step=0.1, 
                        key=k('oldpeak'), value=st.session_state[k('oldpeak')], placeholder="Ex: 1.0")
        
        curr_ves = st.session_state.get(k('noofmajorvessels'))
        st.selectbox(get_label("Nº Vasos Principais", k('noofmajorvessels')), [0, 1, 2, 3], 
                     index=curr_ves, key=k('noofmajorvessels'), placeholder="Selecione...")
        
        opts_ang = list(MAP_ANGINA.keys())
        curr_ang = st.session_state.get(k('exerciseangia'))
        idx_ang = opts_ang.index(curr_ang) if curr_ang in opts_ang else None
        st.selectbox(get_label("Angina por Exercício", k('exerciseangia')), opts_ang, index=idx_ang, key=k('exerciseangia'), placeholder="Selecione...")

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        opts_cp = list(MAP_CHEST_PAIN.keys())
        curr_cp = st.session_state.get(k('chestpain'))
        idx_cp = opts_cp.index(curr_cp) if curr_cp in opts_cp else None
        st.selectbox(get_label("Tipo de Dor no Peito", k('chestpain')), opts_cp, index=idx_cp, key=k('chestpain'), placeholder="Selecione...")
        
        opts_ecg = list(MAP_ECG.keys())
        curr_ecg = st.session_state.get(k('restingrelectro'))
        idx_ecg = opts_ecg.index(curr_ecg) if curr_ecg in opts_ecg else None
        st.selectbox(get_label("ECG em Repouso", k('restingrelectro')), opts_ecg, index=idx_ecg, key=k('restingrelectro'), placeholder="Selecione...")
    
    with col4:
        opts_slp = ["Ascendente", "Plano", "Descendente", "Não informado"]
        curr_slp = st.session_state.get(k('slope'))
        idx_slp = opts_slp.index(curr_slp) if curr_slp in opts_slp else None
        st.selectbox(get_label("Inclinação ST (Slope)", k('slope')), opts_slp, index=idx_slp, key=k('slope'), placeholder="Selecione...")

    raw_fbs = st.session_state[k('fastingbloodsugar')]
    binary_fbs = (1 if raw_fbs > 120 else 0) if raw_fbs is not None else None

    return {
        'age': st.session_state[k('age')],
        'gender': MAP_GENDER.get(st.session_state[k('gender')]),
        'restingBP': st.session_state[k('restingBP')],
        'serumcholestrol': st.session_state[k('serumcholestrol')],
        'fastingbloodsugar': binary_fbs,
        'maxheartrate': st.session_state[k('maxheartrate')],
        'oldpeak': st.session_state[k('oldpeak')],
        'noofmajorvessels': st.session_state[k('noofmajorvessels')],
        'exerciseangia': MAP_ANGINA.get(st.session_state[k('exerciseangia')]),
        'chestpain': MAP_CHEST_PAIN.get(st.session_state[k('chestpain')]),
        'restingrelectro': MAP_ECG.get(st.session_state[k('restingrelectro')]),
        'slope': MAP_SLOPE.get(st.session_state[k('slope')])
    }

# Lógica principal da aplicação
def main():
    st.markdown("""
        <style>
        .main-title {
            text-align: center; font-size: 2.2rem; font-weight: 700;
            background: linear-gradient(135deg, #ff4b4b 0%, #ff6b6b 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        </style>
        <div class="main-title">❤️ Sistema de Apoio ao Diagnóstico</div>
    """, unsafe_allow_html=True)

    model = load_model()
    scaler = load_scaler()

    mode = st.radio("Selecione o modo:", ["📝 Inserção Manual", "📂 Upload de Exames (PDF)"], 
                    index=0 if st.session_state.view_mode == "📝 Inserção Manual" else 1, horizontal=True)
    
    if mode != st.session_state.view_mode:
        st.session_state.view_mode = mode
        st.rerun()

    if st.session_state.view_mode == "📝 Inserção Manual":
        st.caption("Preencha manualmente.")
        initialize_form_state({}, "manual")
        data = render_form("manual")
        
        if st.button("🔍 Analisar", type="primary"):
            if any(v is None for v in data.values()):
                st.error("Preencha os campos em vermelho.")
            elif model:
                try:
                    res = make_prediction(model, scaler, data)
                    show_result(res)
                except Exception as e: st.error(f"Erro: {e}")

    elif st.session_state.view_mode == "📂 Upload de Exames (PDF)":
        uploaded_files = st.file_uploader("PDFs", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("🚀 Processar PDF"):
                try:
                    extracted = extract_features_from_pdfs(uploaded_files)
                    for k in extracted.keys():
                        key = f"pdf_{k}"
                        if key in st.session_state: del st.session_state[key]
                    
                    initialize_form_state(extracted, "pdf")
                    st.session_state.pdf_loaded = True
                    st.rerun()
                except Exception as e: st.error(f"Erro: {e}")

        if st.session_state.get('pdf_loaded'):
            st.divider()
            initialize_form_state({}, "pdf")
            data_pdf = render_form("pdf")
            
            col_b1, col_b2 = st.columns([1, 0.3])
            with col_b1:
                if st.button("🔍 Analisar Dados PDF", type="primary"):
                    if any(v is None for v in data_pdf.values()):
                        st.error("Preencha os campos em vermelho.")
                    elif model:
                        try:
                            res = make_prediction(model, scaler, data_pdf)
                            show_result(res)
                        except Exception as e: st.error(f"Erro: {e}")
            with col_b2:
                if st.button("Limpar"):
                    st.session_state.pdf_loaded = False
                    st.rerun()

# Exibição do resultado final
def show_result(prediction):
    st.markdown("### Resultado:")
    if prediction == 1: st.error("⚠️ ALTO RISCO DETECTADO")
    else: st.success("✅ BAIXO RISCO DETECTADO")

if __name__ == "__main__":
    main()