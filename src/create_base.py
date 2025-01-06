import os
import logging
import platform
import torch
import shutil
from typing import List
from pathlib import Path
from haystack.document_stores import FAISSDocumentStore  # Importation corrigée
from haystack.nodes import DensePassageRetriever
from haystack.schema import Document
from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
console = Console()
logger = logging.getLogger("rich")

# Détection de l'environnement Mac M1/M2
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

def is_valid_file(file_path: str) -> bool:
    """
    Vérifie si un fichier est valide pour le traitement.
    
    Args:
        file_path: Chemin du fichier à vérifier
        
    Returns:
        bool: True si le fichier est valide, False sinon
    """
    # Extensions de fichiers supportées
    VALID_EXTENSIONS = {'.txt', '.md', '.pdf', '.doc', '.docx', '.rtf'}
    
    # Fichiers à ignorer
    IGNORE_FILES = {'.DS_Store', 'thumbs.db', '.git'}
    
    file_name = os.path.basename(file_path)
    extension = os.path.splitext(file_path)[1].lower()
    
    # Vérifier si le fichier doit être ignoré
    if file_name in IGNORE_FILES:
        return False
    
    # Vérifier si l'extension est supportée
    if extension not in VALID_EXTENSIONS:
        return False
    
    # Vérifier si le fichier est caché
    if file_name.startswith('.'):
        return False
    
    return True

def load_documents(directory: str) -> List[Document]:
    """
    Charge les documents depuis un répertoire.
    
    Args:
        directory: Chemin du répertoire contenant les documents
        
    Returns:
        List[Document]: Liste des documents chargés
    """
    documents = []
    directory_path = Path(directory)
    
    try:
        # Parcourir récursivement le répertoire
        for file_path in tqdm(list(directory_path.rglob('*')), desc="Chargement des documents"):
            if not file_path.is_file() or not is_valid_file(str(file_path)):
                continue
            
            try:
                # Lire le contenu du fichier
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Créer un document Haystack
                doc = Document(
                    content=content,
                    meta={
                        'name': file_path.name,
                        'path': str(file_path),
                        'type': file_path.suffix[1:] if file_path.suffix else 'unknown'
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de {file_path}: {str(e)}")
                continue
    
    except Exception as e:
        logger.error(f"Erreur lors du parcours du répertoire: {str(e)}")
        raise
    
    logger.info(f"Nombre de documents chargés: {len(documents)}")
    return documents

def create_vector_store(input_dir: str, output_dir: str) -> None:
    """
    Crée un vector store à partir des documents.
    
    Args:
        input_dir: Répertoire contenant les documents
        output_dir: Répertoire de sortie pour l'index
    """
    try:
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Définir les chemins
        faiss_index_path = os.path.join(output_dir, "faiss_index")
        db_path = os.path.join(output_dir, "faiss_document_store.db")
        
        # Supprimer les anciens fichiers s'ils existent
        if os.path.exists(faiss_index_path):
            shutil.rmtree(faiss_index_path)
        if os.path.exists(db_path):
            os.remove(db_path)
            
        # Créer un nouveau document store
        document_store = FAISSDocumentStore(
            sql_url=f"sqlite:///{db_path}",
            embedding_dim=384,
            return_embedding=True,
            similarity="cosine",
            faiss_index_factory_str="Flat"
        )
        
        # Charger les documents
        docs = load_documents(input_dir)
        if not docs:
            raise ValueError("Aucun document valide trouvé dans le répertoire spécifié")
            
        # Écrire les documents dans la base
        logger.info("Écriture des documents dans la base...")
        document_store.write_documents(docs)
        
        # Créer le retriever
        retriever = DensePassageRetriever(
            document_store=document_store,
            query_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            passage_embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            use_gpu=DEVICE != "cpu",
            batch_size=16 if IS_MAC_ARM else 32,
            embed_title=True,
            use_fast_tokenizers=True
        )
        
        # Mettre à jour les embeddings
        logger.info("Calcul et mise à jour des embeddings...")
        document_store.update_embeddings(retriever)
        
        # Sauvegarder l'index
        logger.info("Sauvegarde de l'index FAISS...")
        document_store.save(faiss_index_path)
        logger.info(f"Vector store créé avec succès dans {faiss_index_path}")
        logger.info(f"Base de données SQLite créée dans {db_path}")
        
        # Vérifier la synchronisation
        doc_count = document_store.get_document_count()
        embedding_count = document_store.get_embedding_count()
        logger.info(f"Documents dans SQLite: {doc_count}")
        logger.info(f"Embeddings dans FAISS: {embedding_count}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la création du vector store: {str(e)}")
        raise

def main():
    """Fonction principale."""
    try:
        # Répertoire contenant les documents à indexer
        input_dir = "/Users/taciclei/Documents"
        
        # Répertoire de sortie pour l'index
        output_dir = "output"
        
        # Créer le vector store
        create_vector_store(input_dir, output_dir)
        
    except Exception as e:
        logger.error(f"Erreur dans le programme principal: {str(e)}")
        raise

if __name__ == "__main__":
    main()