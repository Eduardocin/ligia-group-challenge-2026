#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ligia-group-challenge-2026
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	
	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Download dataset do Kaggle
.PHONY: download_data
download_data:
	$(PYTHON_INTERPRETER) src/data_loader.py

## Executar pré-processamento dos dados
.PHONY: preprocess
preprocess:
	$(PYTHON_INTERPRETER) src/preprocessing.py

## Treinar modelo
.PHONY: train
train:
	$(PYTHON_INTERPRETER) src/model_training.py

## Executar fluxo completo: download -> preprocess -> train
.PHONY: pipeline
pipeline: download_data preprocess train
	@echo ">>> Pipeline completo executado com sucesso!"

## Executar app Streamlit
.PHONY: app
app:
	streamlit run src/app.py

## Executar todos os notebooks em sequência
.PHONY: notebooks
notebooks:
	@echo ">>> Executando notebooks..."
	$(PYTHON_INTERPRETER) -m jupyter nbconvert --to notebook --execute notebooks/1.0_carregamento_dados.ipynb
	$(PYTHON_INTERPRETER) -m jupyter nbconvert --to notebook --execute notebooks/1.1_verificacao_qualidade.ipynb
	$(PYTHON_INTERPRETER) -m jupyter nbconvert --to notebook --execute notebooks/1.2_analise_univariada.ipynb
	$(PYTHON_INTERPRETER) -m jupyter nbconvert --to notebook --execute notebooks/1.3_analise_bivariada.ipynb
	$(PYTHON_INTERPRETER) -m jupyter nbconvert --to notebook --execute notebooks/2.0_limpeza_dados.ipynb
	$(PYTHON_INTERPRETER) -m jupyter nbconvert --to notebook --execute notebooks/2.1_feature_engineering.ipynb
	$(PYTHON_INTERPRETER) -m jupyter nbconvert --to notebook --execute notebooks/3.0_treinamento_do_modelo.ipynb
	$(PYTHON_INTERPRETER) -m jupyter nbconvert --to notebook --execute notebooks/3.1_comparacao_com_baseline.ipynb
	@echo ">>> Notebooks executados com sucesso!"



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
