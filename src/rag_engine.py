import os
import logging
from typing import List, Optional, Dict, Any
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, PromptNode, PromptTemplate
from haystack.pipelines import Pipeline
from haystack.schema import Document
import torch
import platform
from rich.console import Console
from rich.logging import RichHandler
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
console = Console()
logger = logging.getLogger("rich")

# Détection de l'environnement
IS_MAC_ARM = platform.processor() == 'arm' and platform.system() == 'Darwin'
if IS_MAC_ARM:
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Utilisation du device: {DEVICE}")

class RAGEngine:
    """Moteur RAG (Retrieval-Augmented Generation) utilisant Haystack."""

    def __init__(self):
        """Initialise le moteur RAG."""
        self.document_store = None
        self.retriever = None
        self.prompt_node = None
        self.pipeline = None
        
        # Template de prompt optimisé
        self.template = PromptTemplate(
            prompt="""Réponds à la question en te basant uniquement sur le contexte fourni.
            Si l'information n'est pas dans le contexte, dis-le explicitement.
            
            Contexte: {join(documents, delimiter='\n\n', max_length=1000)}
            
            Question: {query}
            
            Réponse concise:""",
            output_parser=None
        )

    def initialize(self, output_dir: str = "output") -> None:
        """
        Initialise les composants du RAG.
        
        Args:
            output_dir: Répertoire contenant l'index FAISS et la base SQLite
        """
        try:
            # Vérification des chemins
            faiss_index_path = os.path.join(output_dir, "faiss_index")
            db_path = os.path.join(output_dir, "faiss_document_store.db")
            
            logger.info(f"Vérification des chemins:")
            logger.info(f"- Index FAISS: {faiss_index_path}")
            logger.info(f"- Base SQLite: {db_path}")
            
            if not os.path.exists(faiss_index_path) or not os.path.exists(db_path):
                raise FileNotFoundError(
                    f"Index FAISS ou base SQLite non trouvés dans {output_dir}"
                )
            
            logger.info("Chargement de l'index FAISS...")
            # Chargement de l'index FAISS
            self.document_store = FAISSDocumentStore.load(
                index_path=faiss_index_path,
                config_path=None  # Pas de fichier de configuration supplémentaire
            )
            logger.info("Index FAISS chargé")
            
            # Initialisation de la base SQLite
            logger.info("Initialisation de la base SQLite...")
            self.document_store.sql_url = f"sqlite:///{db_path}"
            logger.info("Base SQLite initialisée")
            
            # Vérification du contenu après chargement
            doc_count = self.document_store.get_document_count()
            embedding_count = self.document_store.get_embedding_count()
            logger.info(f"État après chargement:")
            logger.info(f"- Documents dans SQLite: {doc_count}")
            logger.info(f"- Embeddings dans FAISS: {embedding_count}")
            
            if doc_count != embedding_count:
                logger.warning(f"Désynchronisation détectée: {doc_count} documents vs {embedding_count} embeddings")
            
            # Initialisation du retriever
            logger.info("Initialisation du retriever...")
            self.retriever = DensePassageRetriever(
                document_store=self.document_store,
                query_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                passage_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                use_gpu=DEVICE != "cpu",
                batch_size=16 if IS_MAC_ARM else 32,
                embed_title=True,
                use_fast_tokenizers=True
            )
            logger.info("Retriever initialisé")
            
            # Initialisation du nœud de prompt
            logger.info("Initialisation du nœud de prompt...")
            self.prompt_node = PromptNode(
                model_name_or_path="google/flan-t5-base",
                default_prompt_template=self.template,
                use_gpu=DEVICE != "cpu",
                max_length=512,
                model_kwargs={
                    "temperature": 0.7,
                    "top_p": 0.95,
                }
            )
            logger.info("Nœud de prompt initialisé")
            
            # Construction du pipeline
            logger.info("Construction du pipeline...")
            self.pipeline = Pipeline()
            self.pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
            self.pipeline.add_node(component=self.prompt_node, name="PromptNode", inputs=["Retriever"])
            logger.info("Pipeline RAG initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du RAG: {str(e)}")
            logger.error(f"Détails de l'erreur:", exc_info=True)
            raise

    def query(self, 
              query: str, 
              top_k: int = 3, 
              filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Exécute une requête sur le pipeline RAG.
        
        Args:
            query: Question à poser
            top_k: Nombre de documents à récupérer
            filters: Filtres optionnels pour la recherche
            
        Returns:
            Dict contenant la réponse et les documents pertinents
        """
        try:
            if not self.pipeline:
                raise ValueError("Pipeline non initialisé. Appelez initialize() d'abord.")
            
            logger.info(f"Exécution de la requête: {query}")
            # Exécution du pipeline
            output = self.pipeline.run(
                query=query,
                params={
                    "Retriever": {
                        "top_k": top_k,
                        "filters": filters
                    },
                    "PromptNode": {
                        "generation_kwargs": {
                            "max_new_tokens": 256,
                            "num_beams": 2,
                            "early_stopping": True
                        }
                    }
                }
            )
            
            # Formatage des documents
            documents = [{
                "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                "score": round(float(doc.score), 4),
                "meta": doc.meta
            } for doc in output["documents"]]
            
            logger.info(f"Nombre de documents trouvés: {len(documents)}")
            
            return {
                "query": query,
                "answer": output["results"][0],
                "documents": documents
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de la requête: {str(e)}")
            logger.error(f"Détails de l'erreur:", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "documents": []
            }

def main():
    """Fonction principale de test."""
    try:
        # Initialisation
        logger.info("Démarrage du moteur RAG...")
        rag = RAGEngine()
        rag.initialize()
        
        # Test avec une requête
        query = "Que contiennent mes documents ?"
        logger.info(f"Test avec la requête: {query}")
        result = rag.query(query)
        
        # Affichage des résultats
        console.print("\n[bold blue]Question:[/bold blue]", query)
        console.print("\n[bold green]Réponse:[/bold green]", result["answer"])
        console.print("\n[bold yellow]Documents pertinents:[/bold yellow]")
        
        for doc in result["documents"]:
            console.print(f"\n- [bold]{doc['meta'].get('name', 'Sans nom')}[/bold]")
            console.print(f"  Score: {doc['score']}")
            console.print(f"  Extrait: {doc['content'][:200]}...")
            
    except Exception as e:
        logger.error(f"Erreur dans le programme principal: {str(e)}")
        logger.error(f"Détails de l'erreur:", exc_info=True)

if __name__ == "__main__":
    main()