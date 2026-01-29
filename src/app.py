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
import shap
from model_training import run_model_training_pipeline
# Configurações iniciais
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'models'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

st.set_page_config(page_title="Diagnóstico Cardiovascular", page_icon="❤️", layout="wide")

# Dicionários centralizados de conversão e categorização
CATEGORY_MAPS = {
    'gender': {
        'input': {"Masculino": 1, "Feminino": 0},
        'output': {0: 'Feminino', 1: 'Masculino'}
    },
    'exerciseangia': {
        'input': {"Sim": 1, "Não": 0},
        'output': {0: 'Não', 1: 'Sim'}
    },
    'chestpain': {
        'input': {"Angina Típica": 0, "Angina Atípica": 1, "Dor Não-Anginosa": 2, "Assintomático": 3},
        'output': {0: 'Angina Típica', 1: 'Angina Atípica', 2: 'Dor Não-Anginosa', 3: 'Assintomático'}
    },
    'restingrelectro': {
        'input': {"Normal": 0, "Anormalidade ST-T": 1, "Hipertrofia Ventricular": 2},
        'output': {0: 'Normal', 1: 'Anormalidade ST-T', 2: 'Hipertrofia Ventricular'}
    },
    'slope': {
        'input': {"Ascendente": 1, "Plano": 2, "Descendente": 3, "Não informado": None},
        'output': {1: 'Ascendente', 2: 'Plano', 3: 'Descendente'}
    },
    'age_group': {
        'output': {0: '<40 anos', 1: '40-60 anos', 2: '>60 anos'}
    },
    'chol_category': {
        'output': {0: 'Normal (<200)', 1: 'Limítrofe (200-240)', 2: 'Alto (>240)'}
    },
    'bp_category': {
        'output': {0: 'Normal (<120)', 1: 'Elevada (120-160)', 2: 'Alta (>160)'}
    }
}

# Legacy maps para compatibilidade
MAP_GENDER = CATEGORY_MAPS['gender']['input']
MAP_ANGINA = CATEGORY_MAPS['exerciseangia']['input']
MAP_CHEST_PAIN = CATEGORY_MAPS['chestpain']['input']
MAP_ECG = CATEGORY_MAPS['restingrelectro']['input']
MAP_SLOPE = CATEGORY_MAPS['slope']['input']

# Mapeamento de nomes das features
FEATURE_NAMES = {
    'noofmajorvessels': 'Nº de Vasos Afetados',
    'bp_age_index': 'Índice PA × Idade',
    'cholesterol_age_ratio': 'Colesterol/Idade',
    'maxheartrate': 'Freq. Cardíaca Máxima',
    'chronotropic_reserve': 'Reserva Cronotrópica',
    'oldpeak': 'Depressão de ST',
    'gender': 'Gênero',
    'chestpain': 'Tipo de Dor no Peito',
    'slope': 'Inclinação do ST',
    'restingrelectro': 'ECG em Repouso',
    'age_group': 'Faixa Etária',
    'chol_category': 'Categoria de Colesterol',
    'bp_category': 'Categoria de PA'
}

# Controle de estado da interface
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "📝 Inserção Manual"

# Carregamento do modelo e do scaler
@st.cache_resource
def load_model():
    model_path = MODELS_DIR / 'best_model.pkl'
    if model_path.exists():
        return joblib.load(model_path)
    else:
        run_model_training_pipeline()
        return joblib.load(model_path)

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
def make_prediction(model, scaler, data: dict) -> tuple:
    """
    Realiza predição e retorna (resultado, X_processado, importâncias)
    """
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
    
    X = pd.DataFrame([safe_data])
    
    # Aplicar escalonamento nas features numéricas originais
    numerical_features = ['age', 'restingBP', 'serumcholestrol', 
                          'maxheartrate', 'oldpeak', 'noofmajorvessels', 
                          'cholesterol_age_ratio', 'bp_age_index', 'chronotropic_reserve']
    
    if scaler: 
        X[numerical_features] = scaler.transform(X[numerical_features])

    # Features esperadas pelo modelo (sem as originais age, restingBP, serumcholestrol, fastingbloodsugar, exerciseangia)
    model_features = ['noofmajorvessels', 'bp_age_index', 'cholesterol_age_ratio', 
                      'maxheartrate', 'chronotropic_reserve', 'oldpeak', 
                      'gender', 'chestpain', 'restingrelectro', 'slope', 
                      'age_group', 'chol_category', 'bp_category']
    
    # Selecionar apenas as features que o modelo espera
    X = X[model_features]
    
    prediction = int(model.predict(X)[0])
    
    # Calcular SHAP values (importância LOCAL para esta predição específica)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Para classificação binária, pegar os valores para a classe positiva (doença)
        if isinstance(shap_values, list):
            shap_values_local = shap_values[1][0]  # Classe 1 (doença)
        else:
            shap_values_local = shap_values[0]
    except Exception as e:
        print(f"Erro ao calcular SHAP: {e}")
        shap_values_local = None
    
    return prediction, X, model_features, shap_values_local

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

# Carregar CSS externo
def load_css():
    css_file = Path(__file__).parent / 'static' / 'style.css'
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Lógica principal da aplicação
def main():
    load_css()
    
    # Header personalizado
    st.markdown("""
        <div class="custom-header">
            <div class="header-title">
                <div class="header-icon">❤️</div>
                <div class="header-text">Sistema de Apoio ao Diagnóstico</div>
                <div class="header-icon">❤️</div>
            </div>
            <div class="header-subtitle">
                Análise preditiva para doenças cardiovasculares
            </div>
            <div class="features-grid">
                <div class="feature-badge">
                    <span class="feature-icon">🩺</span>
                    <span>IA Avançada</span>
                </div>
                <div class="feature-badge">
                    <span class="feature-icon">📊</span>
                    <span>Análise Precisa</span>
                </div>
                <div class="feature-badge">
                    <span class="feature-icon">⚡</span>
                    <span>Resultado Rápido</span>
                </div>
            </div>
        </div>
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
                st.error("Preencha todos os campos.")
            elif model:
                try:
                    res, X_data, features, shap_vals = make_prediction(model, scaler, data)
                    show_result(res, X_data, features, shap_vals, scaler)
                except Exception as e: st.error(f"Erro: {e}")

    elif st.session_state.view_mode == "📂 Upload de Exames (PDF)":
        uploaded_files = st.file_uploader("📄 Enviar Exames em PDF", type="pdf", accept_multiple_files=True, label_visibility="visible")
        
        if uploaded_files:
            if st.button("🚀 Processar PDF", type="primary"):
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
                            res, X_data, features, shap_vals = make_prediction(model, scaler, data_pdf)
                            show_result(res, X_data, features, shap_vals, scaler)
                        except Exception as e: st.error(f"Erro: {e}")
            with col_b2:
                if st.button("🗑️ Limpar", type="primary"):
                    st.session_state.pdf_loaded = False
                    st.rerun()



# Features numéricas que precisam ser "desnormalizadas" para visualização
NUMERICAL_FEATURES_ORIGINAL = [
    'age', 'restingBP', 'serumcholestrol', 'maxheartrate', 
    'oldpeak', 'noofmajorvessels', 'cholesterol_age_ratio', 
    'bp_age_index', 'chronotropic_reserve'
]

def desnormalizar_valor(feature_name, valor_normalizado, scaler):
    #Desnormaliza um valor numérico usando o scaler
    if scaler is None or feature_name not in NUMERICAL_FEATURES_ORIGINAL:
        return valor_normalizado
    
    try:
        if hasattr(scaler, 'feature_names_in_'):
            if feature_name not in scaler.feature_names_in_:
                return valor_normalizado
            idx = list(scaler.feature_names_in_).index(feature_name)
        else:
            return valor_normalizado
        
        valor_original = valor_normalizado * scaler.scale_[idx] + scaler.mean_[idx]
        return valor_original
    except:
        return valor_normalizado

def formatar_valor_feature(feature_name, valor):
    # Tentar converter para int para features categóricas
    valor_int = int(round(valor))
    
    if feature_name in CATEGORY_MAPS and 'output' in CATEGORY_MAPS[feature_name]:
        return CATEGORY_MAPS[feature_name]['output'].get(valor_int, f"{valor_int}")
    
    # Features numéricas - se for inteiro, mostrar sem decimais
    if valor == valor_int:
        return f"{valor_int}"
    
    # Caso contrário, mostrar com 1-2 casas decimais
    if abs(valor) < 10:
        return f"{valor:.2f}"
    else:
        return f"{valor:.1f}"

# Exibição do resultado final
def show_result(prediction, X_data=None, feature_names=None, shap_values_local=None, scaler=None):
    st.markdown("### 📋 Resultado da Análise:")
    
    # Resultado principal
    col1, col2 = st.columns([2, 1])
    with col1:
        if prediction == 1:
            st.error("⚠️ ALTO RISCO DE DOENÇA CARDIOVASCULAR DETECTADO", icon="🚨")
        else:
            st.success("✅ BAIXO RISCO DETECTADO", icon="✅")
    
    # Mostrar importância LOCAL das features se disponível (SHAP values)
    if X_data is not None and feature_names is not None and shap_values_local is not None:
        st.divider()
        st.markdown("#### 🔍 Fatores Decisivos Para ESTA Análise:")
        st.caption("Cada análise pode ter influenciadores DIFERENTES, dependendo dos valores do paciente")
        
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Value': shap_values_local,
            'Abs_SHAP': np.abs(shap_values_local),
            'Valor_Paciente': X_data.iloc[0].values
        }).sort_values('Abs_SHAP', ascending=False) 
        
        # Top 5 features que mais influenciaram ESTA predição
        top_features = shap_df.head(5)
        
        # Exibir gráfico com cores (vermelho = aumenta risco, verde = diminui risco)
        col_chart, col_table = st.columns([1.5, 1])
        
        with col_chart:
            chart_data = top_features.copy()
            chart_data['Feature_PT'] = chart_data['Feature'].map(FEATURE_NAMES)
            chart_data = chart_data.sort_values('Abs_SHAP', ascending=False)  
            
            st.markdown("**Impacto de Cada Fator:**")
            
            max_impact = chart_data['Abs_SHAP'].max()
            
            for _, row in chart_data.iterrows():
                feature_name = row['Feature_PT']
                shap_val = row['SHAP_Value']
                abs_val = abs(shap_val)
                
                # Calcular porcentagem
                pct_width = int((abs_val / max_impact * 100)) if max_impact > 0 else 0
                pct_impact = abs(shap_val) * 100 
                
                # Cor baseada no sinal
                color = '#51CF66' if shap_val < 0 else '#FF6B6B'
                direction = '↓ Reduz' if shap_val < 0 else '↑ Aumenta'
                
                # HTML para barra colorida com porcentagem
                html_bar = f"""
                <div style="margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="font-weight: bold; font-size: 14px;">{feature_name}</span>
                        <span style="font-size: 12px; color: {'#51CF66' if shap_val < 0 else '#FF6B6B'}">{direction} risco ({pct_impact:.1f}%)</span>
                    </div>
                    <div style="width: 100%; background-color: #e0e0e0; border-radius: 4px; height: 20px; overflow: hidden;">
                        <div style="width: {pct_width}%; background-color: {color}; height: 100%; transition: width 0.3s;"></div>
                    </div>
                </div>
                """
                st.markdown(html_bar, unsafe_allow_html=True)
            
            st.caption("""
            🟢 **Verde**: fatores que REDUZEM o risco | 🔴 **Vermelho**: fatores que AUMENTAM o risco
            """)
        
        with col_table:
            st.markdown("**Top 5 Influenciadores:**")
            for idx, (_, row) in enumerate(top_features.iterrows(), 1):
                feature_pt = FEATURE_NAMES.get(row['Feature'], row['Feature'])
                direction = "↑ Aumenta" if row['SHAP_Value'] > 0 else "↓ Diminui"
                pct_impact = abs(row['SHAP_Value']) * 100
                
                # Tentar desnormalizar o valor
                valor_display = desnormalizar_valor(row['Feature'], row['Valor_Paciente'], scaler)
                valor_formatado = formatar_valor_feature(row['Feature'], valor_display)
                
                st.markdown(f"""
                **{idx}. {feature_pt}**  
                {direction} risco ({pct_impact:.1f}%)  
                Seu valor: {valor_formatado}
                """)
        
        st.divider()
        
        if prediction == 1:
            st.markdown("""
            ⚠️ **Importante - Próximos Passos:**
            
            Os fatores listados acima indicaram um risco potencial de doença cardiovascular. 
            **Isto NÃO é um diagnóstico definitivo** - é apenas um indicador baseado em seus dados clínicos.
            
            ✓ **Recomendações:**
            - Agende uma consulta com um cardiologista para avaliação completa
            - Não ignore estes resultados - a detecção precoce é importante
            - Leve este relatório para sua consulta médica
            - Mantenha hábitos saudáveis: exercício regular, dieta balanceada, redução de estresse
            """)
        else:
            st.markdown("""
            ✅ **Boas Notícias!**
            
            Sua análise não apresentou sinais claros de doença cardiovascular nos fatores avaliados.
            **Porém, isto não substitui uma avaliação médica profissional.**
            
            ✓ **Mantenha:**
            - Exercícios físicos regulares
            - Dieta saudável com pouco sal e gordura
            - Controle do estresse
            - Acompanhamento médico periódico (mesmo sem risco)
            """)


if __name__ == "__main__":
    main()