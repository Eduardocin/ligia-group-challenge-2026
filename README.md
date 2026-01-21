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
python -c "import numpy, pandas, sklearn; print('✅ Ambiente configurado!')"
```



## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         Ligia Group and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── main              <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes main a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## 🛠️ Comandos Úteis (Makefile)

```bash
make requirements    # Instala dependências
make clean          # Remove arquivos compilados Python
make lint           # Verifica qualidade do código
make format         # Formata código automaticamente
```

## 📝 Convenções

- **Notebooks**: Use numeração e descrição, ex: `01-analise-exploratoria.ipynb`
- **Commits**: Siga [Conventional Commits](https://www.conventionalcommits.org/)
- **Código**: Formatado automaticamente com ruff

--------

