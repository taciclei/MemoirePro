# Variables
PYTHON := python
SRC_DIR := src
OUTPUT_DIR := output
DOCS_DIR := /Users/taciclei/Documents
ENV_FILE := .env

# Couleurs pour les messages
BLUE := \033[34m
GREEN := \033[32m
RED := \033[31m
RESET := \033[0m

# Cibles principales
.PHONY: all clean setup run download-model create-base query help

all: setup download-model create-base

help:
	@echo "$(BLUE)Commandes disponibles:$(RESET)"
	@echo "  $(GREEN)make setup$(RESET)         - Prépare l'environnement"
	@echo "  $(GREEN)make clean$(RESET)         - Nettoie les fichiers générés"
	@echo "  $(GREEN)make all$(RESET)           - Execute toute la chaîne de traitement"
	@echo "  $(GREEN)make download-model$(RESET) - Télécharge le modèle"
	@echo "  $(GREEN)make create-base$(RESET)   - Crée la base vectorielle"
	@echo "  $(GREEN)make query$(RESET)         - Lance le moteur de requêtes"

# Vérifie si le fichier .env existe
check-env:
	@if [ ! -f $(ENV_FILE) ]; then \
		echo "$(RED)Erreur: Fichier .env manquant$(RESET)"; \
		echo "Créez un fichier .env avec HUGGINGFACE_API_KEY=votre_clé"; \
		exit 1; \
	fi

# Crée les répertoires nécessaires
setup: check-env
	@echo "$(BLUE)Configuration de l'environnement...$(RESET)"
	@mkdir -p $(OUTPUT_DIR)
	@touch $(OUTPUT_DIR)/.gitkeep

# Nettoie les fichiers générés
clean:
	@echo "$(BLUE)Nettoyage des fichiers générés...$(RESET)"
	@rm -rf $(OUTPUT_DIR)/faiss_index
	@rm -f $(OUTPUT_DIR)/faiss_document_store.db
	@rm -f $(OUTPUT_DIR)/vector_store.txt
	@mkdir -p $(OUTPUT_DIR)
	@touch $(OUTPUT_DIR)/.gitkeep

# Télécharge le modèle
download-model: check-env
	@echo "$(BLUE)Téléchargement du modèle...$(RESET)"
	@$(PYTHON) $(SRC_DIR)/download_model.py

# Crée la base vectorielle
create-base: check-env clean
	@echo "$(BLUE)Création de la base vectorielle...$(RESET)"
	@$(PYTHON) $(SRC_DIR)/create_base.py
	@if [ $$? -eq 0 ]; then \
		echo "$(GREEN)Base vectorielle créée avec succès$(RESET)"; \
	else \
		echo "$(RED)Erreur lors de la création de la base vectorielle$(RESET)"; \
		exit 1; \
	fi

# Lance le moteur de requêtes
query: check-env
	@echo "$(BLUE)Lancement du moteur de requêtes...$(RESET)"
	@$(PYTHON) $(SRC_DIR)/rag_engine.py

# Vérifie l'état de l'index
status:
	@echo "$(BLUE)État de l'index:$(RESET)"
	@if [ -d "$(OUTPUT_DIR)/faiss_index" ]; then \
		echo "$(GREEN)Index FAISS: Présent$(RESET)"; \
	else \
		echo "$(RED)Index FAISS: Manquant$(RESET)"; \
	fi
	@if [ -f "$(OUTPUT_DIR)/faiss_document_store.db" ]; then \
		echo "$(GREEN)Base SQLite: Présente$(RESET)"; \
	else \
		echo "$(RED)Base SQLite: Manquante$(RESET)"; \
	fi