# Ligia Group Challenge 2026

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Projeto de Data Science para análise e predição de doenças cardiovasculares

## 📊 Dataset

Este projeto utiliza o **Cardiovascular Disease Dataset** do Kaggle, que contém 1000 registros de pacientes com 14 variáveis relacionadas a fatores de risco cardiovascular.

### Variáveis do Dataset:
- **patientid**: ID do paciente
- **age**: Idade
- **gender**: Gênero (0=Feminino, 1=Masculino)
- **chestpain**: Tipo de dor no peito
- **restingBP**: Pressão arterial em repouso
- **serumcholestrol**: Colesterol sérico
- **fastingbloodsugar**: Glicemia em jejum
- **restingrelectro**: Resultados do eletrocardiograma em repouso
- **maxheartrate**: Frequência cardíaca máxima
- **exerciseangia**: Angina induzida por exercício
- **oldpeak**: Depressão do segmento ST
- **slope**: Inclinação do segmento ST
- **noofmajorvessels**: Número de vasos principais
- **target**: Presença de doença cardíaca (0=Não, 1=Sim)

## 🚀 Setup do Ambiente

### Pré-requisitos
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou Anaconda instalado
- Git

### Instalação

1. **Clone o repositório**
   ```bash
   git clone <url-do-repositorio>
   cd ligia-group-challenge-2026
   ```

2. **Crie o ambiente conda**
   ```bash
   conda create -n ligia python=3.12 -y
   ```

3. **Ative o ambiente**
   ```bash
   conda activate ligia
   ```

4. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure o kernel do Jupyter** (opcional, para usar notebooks)
   ```bash
   python -m ipykernel install --user --name=ligia --display-name="Python (ligia)"
   ```

### Verificação da Instalação

```bash
# Verifique se o ambiente está ativo
conda info --envs

# Teste a instalação de pacotes
python -c "import numpy, pandas, sklearn, mlcroissant; print('✅ Ambiente configurado!')"
```

## 🔄 Fluxo de Execução

### Opção 1: Pipeline Completo Automatizado (Recomendado)

```bash
make pipeline
```

Este comando executa automaticamente:
1. **Download dos dados** (`make download_data`) - Baixa o dataset do Kaggle via ML Croissant
2. **Pré-processamento** (`make preprocess`) - Limpa dados e cria features
3. **Treinamento** (`make train`) - Treina e salva o modelo

### Opção 2: Executar Etapas Individualmente

```bash
# 1. Baixar dados
make download_data

# 2. Pré-processar dados
make preprocess

# 3. Treinar modelo
make train

# 4. Executar app
make app
```

### Opção 3: Executar Scripts Python Diretamente

```bash
# 1. Baixar dados
python src/data_loader.py

# 2. Pré-processar dados
python src/preprocessing.py

# 3. Treinar modelo
python src/model_training.py

# 4. Executar app
streamlit run src/app.py
```

## 📊 Como Executar o App Streamlit

```bash
make app
# Ou
streamlit run src/app.py
```

Abra o navegador em `http://localhost:8501` para visualizar o app.

## 📓 Executar Notebooks

### Opção 1: Jupyter Lab
```bash
jupyter lab
```

### Opção 2: VS Code
Abra os notebooks diretamente no VS Code com a extensão Jupyter instalada.

### Opção 3: Executar todos os notebooks via CLI
```bash
make notebooks
```

## 🛠️ Comandos Úteis do Makefile

```bash
make help              # Ver todos os comandos disponíveis
make requirements      # Instalar/atualizar dependências
make clean             # Limpar arquivos compilados Python
make lint              # Verificar código com ruff
make format            # Formatar código com ruff
make create_environment # Criar ambiente conda
```


## 📁 Project Organization

```
├── LICENSE                    <- Licença open-source do projeto
├── Makefile                   <- Automação com comandos: make pipeline, make app, etc.
├── README.md                  <- Documentação principal do projeto
├── pyproject.toml             <- Configuração do projeto e metadados do pacote
├── requirements.txt           <- Dependências Python (gerado com pip freeze)
│
├── data/                      <- Dados do projeto (não versionados no Git)
│   ├── external/              <- Dados de fontes externas
│   ├── interim/               <- Dados intermediários transformados
│   │   └── Cardiovascular_Disease_Dataset_Clean.csv
│   ├── processed/             <- Datasets finais para modelagem
│   │   ├── X_train.csv        <- Features de treino
│   │   ├── X_test.csv         <- Features de teste
│   │   ├── y_train.csv        <- Target de treino
│   │   ├── y_test.csv         <- Target de teste
│   │   ├── scaler.pkl         <- Scaler treinado
│   │   └── selected_features.pkl <- Features selecionadas
│   └── raw/                   <- Dados originais imutáveis
│       └── Cardiovascular_Disease_Dataset.csv
│
├── dados_exames/              <- Dados de exames médicos (PDFs)
│
├── docs/                      <- Documentação do projeto
│
├── models/                    <- Modelos treinados e serializados (.pkl)
│
├── notebooks/                 <- Jupyter notebooks para análise exploratória
│   ├── 1.0_carregamento_dados.ipynb      <- Download e carregamento inicial
│   ├── 1.1_verificacao_qualidade.ipynb   <- Verificação de qualidade dos dados
│   ├── 1.2_analise_univariada.ipynb      <- Análise de variáveis individuais
│   ├── 1.3_analise_bivariada.ipynb       <- Análise de relações entre variáveis
│   ├── 2.0_limpeza_dados.ipynb           <- Limpeza e tratamento de dados
│   ├── 2.1_feature_engineering.ipynb     <- Criação de features
│   ├── 3.1_treinamento_do_modelo.ipynb   <- Treinamento de modelos
│   └── 3.2_comparação_com_baseline.ipynb <- Comparação de modelos
│
├── references/                <- Dicionários de dados, manuais e materiais explicativos
│
├── reports/                   <- Análises geradas (HTML, PDF, etc.)
│   └── figures/               <- Gráficos e figuras geradas
│
└── src/                       <- Código fonte do projeto
    ├── __init__.py            <- Torna src um módulo Python
    ├── data_loader.py         <- Download de dados via ML Croissant
    ├── preprocessing.py       <- Pipeline de pré-processamento e feature engineering
    ├── model_training.py      <- Treinamento e avaliação de modelos
    ├── app.py                 <- Aplicação Streamlit para predições
    ├── teste_extrção_pdf_med.py <- Extração de dados de PDFs médicos
    └── static/                <- Arquivos estáticos para o app
        └── style.css          <- Estilos CSS para Streamlit
```

## 🛠️ Comandos Úteis (Makefile)

```bash
# Pipeline completo
make pipeline           # Executa: download → preprocess → train

# Etapas individuais
make download_data      # Baixa dados do Kaggle via ML Croissant
make preprocess         # Pré-processa dados e cria features
make train              # Treina modelo de classificação
make app                # Inicia aplicação Streamlit

# Notebooks
make notebooks          # Executa todos os notebooks em sequência

# Desenvolvimento
make requirements       # Instala/atualiza dependências
make clean              # Remove arquivos compilados Python
make lint               # Verifica qualidade do código com ruff
make format             # Formata código automaticamente com ruff
make create_environment # Cria ambiente conda
make help               # Lista todos os comandos disponíveis
```

## 📝 Convenções do Projeto

- **Notebooks**: Numeração sequencial `X.Y_descricao.ipynb`
  - `1.x` - Análise exploratória
  - `2.x` - Preparação de dados
  - `3.x` - Modelagem
- **Commits**: Seguem [Conventional Commits](https://www.conventionalcommits.org/)
  - `feat:` - Nova funcionalidade
  - `fix:` - Correção de bug
  - `docs:` - Documentação
  - `style:` - Formatação
  - `refactor:` - Refatoração
- **Código**: Formatado automaticamente com ruff
- **Dados**: Não versionados no Git (`.gitignore`)

--------

