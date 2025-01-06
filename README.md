# MémoirePro

## Description
MémoirePro est un système RAG (Retrieval-Augmented Generation) qui permet d'indexer, rechercher et interroger intelligemment vos documents. Basé sur FAISS pour la recherche vectorielle et FastAPI pour l'interface API, il offre une solution performante pour la gestion documentaire intelligente.

## Prérequis
- Python 3.11
- magic (gestionnaire de paquets)
- Mac M1/M2 (optimisé pour Apple Silicon)

## Installation

1. Cloner le repository :
```bash
git clone <votre-repo>
cd memoirepro
```

2. Installer les dépendances avec magic :
```bash
magic install
```

3. Créer un fichier `.env` à la racine du projet :
```bash
OPENAI_API_KEY=votre_clé_api_openai
```

## Utilisation de l'API

### Démarrer le serveur
```bash
magic shell
uvicorn src.api:app --reload
```
Le serveur démarre sur `http://localhost:8000`

### Endpoints disponibles

1. **Indexer des documents**
```bash
curl -X 'POST' \
  'http://localhost:8000/index' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input_dir": "~/Documents",
  "output_dir": "./output"
}'
```

2. **Charger un index existant**
```bash
curl -X 'POST' \
  'http://localhost:8000/load' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "index_dir": "./output"
}'
```

3. **Poser une question**
```bash
curl -X 'POST' \
  'http://localhost:8000/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "Votre question ici ?",
  "top_k": 3
}'
```

## Fonctionnalités

- Indexation de documents (PDF, DOCX, TXT)
- Recherche sémantique avec FAISS
- Génération de réponses avec GPT-3.5-turbo
- Support optimisé pour Mac M1/M2 (MPS)
- Logging détaillé avec Rich

## Configuration

Le projet utilise [pixi.toml](cci:7://file:///Volumes/r0/haystack/pixi.toml:0:0-0:0) pour la gestion des dépendances. Les principales dépendances incluent :
- farm-haystack avec extras pour la conversion de fichiers
- FastAPI pour l'API REST
- FAISS pour la recherche vectorielle
- SQLAlchemy 1.4.51 pour la persistence

## Contribution
Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.
