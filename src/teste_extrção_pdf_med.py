import pdfplumber
import re
import pandas as pd

def extract_features_from_pdf(pdf_path):
    # Dicionário para armazenar os dados brutos extraídos
    data = {}
    
    # Texto completo para busca global
    full_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            full_text += page.extract_text(layout=True) + "\n"

    # --- 1. Patient ID (Prontuário) ---
    match = re.search(r"Prontuário:\s*(\d+)", full_text)
    data['patientid'] = int(match.group(1)) if match else None

    # --- 2. Age (Idade) ---
    match = re.search(r"\((\d+)\s*anos\)|Idade:\s*(\d+)\s*anos", full_text)
    if match:
        data['age'] = int(match.group(1) if match.group(1) else match.group(2))
    else:
        data['age'] = None

    # --- 3. Gender (Sexo) ---
    # Mapeamento: 0 (female) / 1 (male)
    if re.search(r"\[X\]\s*M", full_text):
        data['gender'] = 1
    elif re.search(r"\[X\]\s*F", full_text):
        data['gender'] = 0
    else:
        data['gender'] = None

    # --- 4. Resting Blood Pressure (PA) ---
    match = re.search(r"PA:\s*(\d+)\s*/\s*(\d+)", full_text)
    data['restingBP'] = int(match.group(1)) if match else None

    # --- 5. Serum Cholesterol ---
    match = re.search(r"COLESTEROL TOTAL\s+(\d+)", full_text)
    data['serumcholestrol'] = int(match.group(1)) if match else None

    # --- 6. Fasting Blood Sugar (Glicose) ---
    # Regra: 1 (true) se > 120 mg/dl, senão 0
    match = re.search(r"GLICOSE \(JEJUM\)\s+(\d+)", full_text)
    if match:
        glucose_val = int(match.group(1))
        data['fastingbloodsugar'] = 1 if glucose_val > 120 else 0
    else:
        data['fastingbloodsugar'] = None

    # --- 7. Chest Pain Type ---
    # 0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic
    if re.search(r"\[X\]\s*Angina Típica", full_text, re.IGNORECASE):
        data['chestpain'] = 0
    elif re.search(r"\[X\]\s*Angina Atípica", full_text, re.IGNORECASE):
        data['chestpain'] = 1
    elif re.search(r"\[X\]\s*Dor Não-Anginosa", full_text, re.IGNORECASE):
        data['chestpain'] = 2
    elif re.search(r"\[X\]\s*Assintomático", full_text, re.IGNORECASE):
        data['chestpain'] = 3
    else:
        data['chestpain'] = None

    # --- 8. Resting Electro Results ---
    # 0: Normal, 1: ST-T abn, 2: LV Hypertrophy
    if re.search(r"Achados do.*?\[X\]\s*Normal", full_text, re.DOTALL):
        data['restingelectro'] = 0
    elif re.search(r"\[X\]\s*Anormalidade de Onda ST-T", full_text):
        data['restingelectro'] = 1
    elif re.search(r"\[X\]\s*Hipertrofia Ventricular", full_text):
        data['restingelectro'] = 2
    else:
        data['restingelectro'] = None

    # --- 9. Max Heart Rate ---
    match = re.search(r"Peak HR\):\s*(\d+)", full_text)
    data['maxheartrate'] = int(match.group(1)) if match else None

    # --- 10. Exercise Induced Angina ---
    if re.search(r"Angina durante esforço:.*?\[X]\s*Sim", full_text, re.DOTALL):
        data['exerciseangina'] = 1
    elif re.search(r"Angina durante esforço:.*?\[X]\s*Não", full_text, re.DOTALL):
        data['exerciseangina'] = 0
    else:
        data['exerciseangina'] = None

    # --- 11. Oldpeak (ST Depression) ---
    match = re.search(r"Depressão de ST:\s*(\d+[.,]?\d*)", full_text)
    if match:
        data['oldpeak'] = float(match.group(1).replace(',', '.'))
    else:
        data['oldpeak'] = None

    # --- 12. Slope of Peak Exercise ST ---
    # 1: Upsloping, 2: Flat (Horizontal), 3: Downsloping
    if re.search(r"Morfologia.*?\[X\]\s*Ascendente", full_text, re.DOTALL):
        data['slope'] = 1
    elif re.search(r"Morfologia.*?\[X\]\s*Horizontal", full_text, re.DOTALL):
        data['slope'] = 2
    elif re.search(r"Morfologia.*?\[X\]\s*Descendente", full_text, re.DOTALL):
        data['slope'] = 3
    else:
        data['slope'] = None

    # --- 13. Number of Major Vessels ---
    match = re.search(r"CONCLUSÃO DO EXAME:.*?Presença de\s*(\d+)", full_text, re.DOTALL)
    data['noofmajorvessels'] = int(match.group(1)) if match else None

    return data

# Caminho do arquivo PDF
pdf_file = "teste_ligia.pdf"

try:
    extracted_data = extract_features_from_pdf(pdf_file)
    
    # Criando o data frame que o modelo vai receber
    df_features = pd.DataFrame([extracted_data])
    cols_to_int = [
        'patientid', 'age', 'gender', 'restingBP', 'chestpain', 'restingelectro', 'maxheartrate', 
        'exerciseangina', 'noofmajorvessels', 'slope'
    ]
    
    for col in cols_to_int:
        df_features[col] = df_features[col].astype('Int64')
    
    print("=== DADOS EXTRAÍDOS PARA O MODELO ===")
    print(df_features.T)
    print(df_features.info())

except Exception as e:
    # Erro caso não for possível ler o pdf
    print(f"Erro ao processar: {e}")