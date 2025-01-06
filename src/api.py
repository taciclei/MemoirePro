import os
import shutil
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

import torch
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, PromptNode, PromptTemplate
from haystack.nodes.file_converter import TextConverter, PDFToTextConverter, DocxToTextConverter
from haystack.pipelines import Pipeline
from haystack.schema import Document

from rich.console import Console
from rich.logging import RichHandler
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1) Chargement des variables d'environnement
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# 2) Configuration du logging avec Rich
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("RAG-API")
console = Console()

# ---------------------------------------------------------------------------
# 3) Configuration de l'API FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG API",
    description="API REST pour un système RAG avec Haystack",
    version="0.2.1"
)

# ---------------------------------------------------------------------------
# 4) Modèles Pydantic pour la validation des entrées
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

    class Config:
        schema_extra = {
            "example": {
                "question": "Quelle est la capitale de la France ?",
                "top_k": 3
            }
        }

class IndexRequest(BaseModel):
    input_dir: str
    output_dir: str

    class Config:
        schema_extra = {
            "example": {
                "input_dir": "~/Documents",
                "output_dir": "./output"
            }
        }

class LoadRequest(BaseModel):
    index_dir: str

    class Config:
        schema_extra = {
            "example": {
                "index_dir": "./output"
            }
        }

# ---------------------------------------------------------------------------
# 5) Détection du device (CPU/GPU/MPS)
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

logger.info(f"Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# 6) Classe RAGEngine
# ---------------------------------------------------------------------------
class RAGEngine:
    """Moteur RAG (Retrieval-Augmented Generation) utilisant Haystack."""

    def __init__(self):
        """Initialise le moteur RAG avec les composants nécessaires."""
        self.document_store = None
        self.retriever = None
        self.prompt_node = None
        self.pipeline = None

        # IMPORTANT : 8 backslashes (\\\\\\\\) pour éviter le SyntaxError
        # Finalement, dans le code source Python, 'delimiter='\\\\\\\\n\\\\\\\\n''
        # Produira assez d'échappements pour le double parsing interne de Haystack
        self.template = PromptTemplate(
            prompt=(
                "Réponds à la question en te basant uniquement sur le contexte fourni.\n"
                "Si l'information n'est pas dans le contexte, dis-le explicitement.\n\n"
                "Contexte:\n{{join(documents, delimiter='\\\\n\\\\n', max_length=1000)}}\n\n"
                "Question: {{query}}\n\n"
                "Réponse concise:"
            )
        )

        logger.debug("RAGEngine instancié.")

    def _load_documents(self, input_dir: str) -> List[Document]:
        """
        Charge et convertit les documents depuis le répertoire d'entrée.
        
        Args:
            input_dir: Chemin vers le répertoire contenant les documents.
        """
        logger.info(f"[LOAD] Chargement des documents depuis {input_dir}")

        # Expansion du chemin utilisateur (~)
        input_dir = os.path.expanduser(input_dir)

        # Converters pour différents formats
        converters = {
            ".txt": TextConverter(),
            ".pdf": PDFToTextConverter(),
            ".docx": DocxToTextConverter()
        }

        documents = []
        input_path = Path(input_dir)

        if not input_path.exists():
            raise ValueError(f"Le répertoire {input_dir} n'existe pas")

        # Parcours récursif des fichiers
        for file_path in input_path.rglob("*"):
            if file_path.suffix.lower() in converters:
                try:
                    logger.debug(f"[LOAD] Conversion de {file_path}")
                    converter = converters[file_path.suffix.lower()]
                    docs = converter.convert(file_path=str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    logger.warning(f"[LOAD] Erreur lors de la conversion de {file_path}: {e}")
                    continue

        return documents

    def index_documents(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Indexe les documents d'un répertoire, puis sauvegarde FAISS + SQLite.
        
        Args:
            input_dir: Répertoire avec les documents.
            output_dir: Répertoire où sauvegarder l'index FAISS et la DB.
        """
        try:
            logger.debug(f"[INDEX] input_dir='{input_dir}', output_dir='{output_dir}'")
            # Expansion et normalisation des chemins
            input_dir = os.path.expanduser(input_dir)
            output_dir = os.path.abspath(output_dir)
            logger.debug(f"[INDEX] Chemins effectifs: input_dir={input_dir}, output_dir={output_dir}")

            os.makedirs(output_dir, exist_ok=True)

            faiss_index_path = os.path.join(output_dir, "faiss_index")
            db_path = os.path.join(output_dir, "faiss_document_store.db")

            # Nettoyage d'anciens fichiers
            if os.path.exists(faiss_index_path):
                logger.debug(f"[INDEX] Suppression ancien index: {faiss_index_path}")
                shutil.rmtree(faiss_index_path)
            if os.path.exists(db_path):
                logger.debug(f"[INDEX] Suppression ancienne DB: {db_path}")
                os.remove(db_path)

            # Création du DocumentStore
            logger.info("[INDEX] Création du FAISSDocumentStore...")
            self.document_store = FAISSDocumentStore(
                sql_url=f"sqlite:///{db_path}",
                embedding_dim=384,
                return_embedding=True,
                similarity="cosine",
                faiss_index_factory_str="Flat"
            )

            # Création du retriever
            logger.info("[INDEX] Création du DensePassageRetriever...")
            self.retriever = DensePassageRetriever(
                document_store=self.document_store,
                query_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                passage_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                use_gpu=(DEVICE != "cpu")
            )

            # Chargement et indexation des documents
            docs = self._load_documents(input_dir)
            if not docs:
                msg = "Aucun document valide trouvé dans le répertoire."
                logger.error("[INDEX] " + msg)
                raise ValueError(msg)

            logger.info(f"[INDEX] Nombre de documents à indexer: {len(docs)}")
            self.document_store.write_documents(docs)

            logger.info("[INDEX] Mise à jour des embeddings...")
            self.document_store.update_embeddings(self.retriever)

            # DEBUG : vérifier doc_count & embedding_count
            doc_count = self.document_store.get_document_count()
            emb_count = self.document_store.get_embedding_count()
            logger.debug(f"[INDEX] Après update_embeddings: doc_count={doc_count}, emb_count={emb_count}")

            logger.info("[INDEX] Sauvegarde de l'index FAISS...")
            self.document_store.save(faiss_index_path)

            logger.info("[INDEX] Indexation terminée.")
            return {
                "status": "success",
                "message": f"{len(docs)} documents indexés avec succès",
                "documents": len(docs),
                "doc_count_in_db": doc_count,
                "embedding_count_in_faiss": emb_count
            }

        except Exception as e:
            logger.error("[INDEX] Erreur lors de la création du vector store", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Erreur lors de l'indexation : {str(e)}"
            )

    def load_index(self, index_dir: str) -> None:
        """
        Charge un index FAISS existant (faiss_index + faiss_document_store.db).
        
        Args:
            index_dir: Chemin du dossier qui contient faiss_index et faiss_document_store.db
        """
        try:
            index_dir = os.path.abspath(index_dir)
            faiss_index_path = os.path.join(index_dir, "faiss_index")
            db_path = os.path.join(index_dir, "faiss_document_store.db")

            if not os.path.exists(faiss_index_path) or not os.path.exists(db_path):
                raise ValueError(
                    f"Index FAISS ou base SQLite manquant dans {index_dir}.\n"
                    "Vérifiez que vous avez bien faiss_index et faiss_document_store.db"
                )

            logger.info(f"[LOAD] Chargement du FAISSDocumentStore (db_path={db_path})")
            self.document_store = FAISSDocumentStore(
                sql_url=f"sqlite:///{db_path}",
                embedding_dim=384,
                similarity="cosine",
                validate_index_sync=True
            )

            logger.info(f"[LOAD] Chargement de l'index FAISS depuis {faiss_index_path}...")
            self.document_store.load(faiss_index_path)

            # DEBUG : vérifier doc_count & embedding_count
            doc_count = self.document_store.get_document_count()
            emb_count = self.document_store.get_embedding_count()
            logger.debug(f"[LOAD] Après load: doc_count={doc_count}, emb_count={emb_count}")

            logger.info("[LOAD] Initialisation du retriever...")
            self.retriever = DensePassageRetriever(
                document_store=self.document_store,
                query_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                passage_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                use_gpu=(DEVICE != "cpu")
            )

            logger.info("[LOAD] Initialisation du PromptNode...")
            self.prompt_node = PromptNode(
                model_name_or_path="gpt-3.5-turbo",
                api_key=os.getenv("OPENAI_API_KEY"),
                default_prompt_template=self.template,
                max_length=2000
            )

            logger.info("[LOAD] Construction du pipeline RAG...")
            self.pipeline = Pipeline()
            self.pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
            self.pipeline.add_node(component=self.prompt_node, name="PromptNode", inputs=["Retriever"])

            logger.info("[LOAD] Chargement terminé avec succès.")

        except Exception as e:
            logger.error("[LOAD] Erreur lors du chargement de l'index", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Erreur lors du chargement : {str(e)}"
            )

    def query(self, question: str, top_k: Optional[int] = 3) -> Dict[str, Any]:
        """
        Interroge le système RAG via un pipeline (Retriever + PromptNode).
        
        Args:
            question: Question à poser.
            top_k: Nombre de documents à récupérer.
        """
        if not self.pipeline:
            raise HTTPException(
                status_code=500,
                detail="Pipeline non initialisé. Appelez d'abord load_index()."
            )

        try:
            logger.info(f"[QUERY] Question: {question} | top_k={top_k}")
            result = self.pipeline.run(
                query=question,
                params={
                    "Retriever": {"top_k": top_k},
                    "PromptNode": {"generation_kwargs": {"max_new_tokens": 200}}
                }
            )

            # On récupère la première réponse
            answers = result.get("answers", [])
            answer = answers[0].answer if answers else ""

            logger.debug(f"[QUERY] answer={answer}")

            return {
                "answer": answer,
                "documents": len(result.get("documents", [])),
                "metadata": {
                    "model": "gpt-3.5-turbo",
                    "top_k": top_k
                }
            }

        except Exception as e:
            logger.error("[QUERY] Erreur lors de la requête", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Erreur lors de la requête : {str(e)}"
            )

# ---------------------------------------------------------------------------
# 7) Instance globale du moteur RAG
# ---------------------------------------------------------------------------
rag_engine = RAGEngine()

# ---------------------------------------------------------------------------
# 8) Endpoints FastAPI
# ---------------------------------------------------------------------------
@app.get("/")
async def root() -> Dict[str, str]:
    """
    Route racine de l'API.
    """
    return {
        "message": "Bienvenue sur l'API RAG",
        "version": "0.2.1",
        "status": "running"
    }

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Endpoint de vérification de santé de l'API.
    """
    return {"status": "healthy"}

@app.post("/index")
async def index_documents(request: IndexRequest) -> Dict[str, Any]:
    """
    Endpoint pour indexer des documents dans FAISS + SQLite.
    """
    return rag_engine.index_documents(request.input_dir, request.output_dir)

@app.post("/load")
async def load_index(request: LoadRequest) -> Dict[str, str]:
    """
    Endpoint pour charger un index FAISS existant.
    """
    rag_engine.load_index(request.index_dir)
    return {"status": "success", "message": "Index chargé avec succès"}

@app.post("/query")
async def query(request: QueryRequest) -> Dict[str, Any]:
    """
    Endpoint pour interroger le pipeline RAG.
    """
    return rag_engine.query(request.question, request.top_k)

# ---------------------------------------------------------------------------
# 9) Point d'entrée du script : uvicorn.run si besoin
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.debug("Lancement de l'application via uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)