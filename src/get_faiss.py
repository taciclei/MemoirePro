import pathlib
from typing import List, Dict
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rag_engine import RAGEngine

console = Console()

class SearchEngine:
    """Moteur de recherche basé sur FAISS."""
    
    def __init__(self, vector_store_path: str, index_path: str):
        """Initialise le moteur de recherche."""
        self.device = "mps" if hasattr(torch, 'mps') and torch.backends.mps.is_available() else "cpu"
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        self.index = faiss.read_index(index_path)
        
        # Charger les documents
        self.documents = []
        with open(vector_store_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    filename, content = line.strip().split("\t", 1)
                    self.documents.append({"filename": filename, "content": content})
                except ValueError:
                    continue

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Recherche les documents les plus pertinents."""
        # Encoder la requête
        query_vector = self.model.encode([query])[0]
        
        # Rechercher les documents similaires
        distances, indices = self.index.search(np.array([query_vector]).astype('float32'), k * 2)
        
        # Filtrer et formater les résultats
        results = []
        seen_contents = set()
        
        for distance, idx in zip(distances[0], indices[0]):
            if idx >= len(self.documents):
                continue
                
            doc = self.documents[idx]
            content = doc["content"]
            
            # Éviter les doublons
            if content in seen_contents:
                continue
            seen_contents.add(content)
            
            # Calculer le score de similarité
            similarity = 1 - (distance / 2)
            if similarity < 0.3:  # Seuil minimal de pertinence
                continue
                
            # Extraire un extrait pertinent
            excerpt = self._get_relevant_excerpt(content, query)
            
            results.append({
                "score": f"{similarity:.1%}",
                "document": doc["filename"],
                "excerpt": excerpt
            })
            
            if len(results) >= k:
                break
                
        return results

    def _get_relevant_excerpt(self, content: str, query: str, context_size: int = 100) -> str:
        """Extrait un passage pertinent du contenu."""
        query_terms = set(query.lower().split())
        content_lower = content.lower()
        
        # Trouver la meilleure position
        best_pos = -1
        max_matches = 0
        
        words = content_lower.split()
        for i in range(len(words)):
            matches = sum(1 for term in query_terms if term in ' '.join(words[i:i+5]))
            if matches > max_matches:
                max_matches = matches
                best_pos = i
        
        if best_pos == -1:
            return content[:200] + "..."
        
        # Extraire le contexte
        start = max(0, best_pos - context_size // 2)
        end = min(len(content), best_pos + context_size // 2)
        excerpt = content[start:end]
        
        # Ajouter des ellipses si nécessaire
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(content):
            excerpt = excerpt + "..."
            
        return excerpt

def main():
    """Fonction principale."""
    vector_store_path = "output/vector_store.txt"
    index_path = "output/faiss_index.index"
    
    try:
        # Initialiser les moteurs de recherche
        search_engine = SearchEngine(vector_store_path, index_path)
        rag_engine = RAGEngine(vector_store_path, index_path)
        
        console.print(Panel.fit(
            "[cyan]Assistant documentaire intelligent[/]\n"
            "1. Mode recherche simple: tapez votre requête\n"
            "2. Mode RAG avec Llama: commencez votre requête par '?'\n"
            "Tapez 'q' pour quitter.",
            title="🔍 Document Assistant"
        ))
        
        while True:
            query = input("\nVotre requête > ").strip()
            if not query:
                continue
            if query.lower() in ('quit', 'exit', 'q'):
                break
            
            # Mode RAG ou recherche simple
            if query.startswith('?'):
                query = query[1:].strip()
                with console.status("[cyan]Analyse en cours..."):
                    result = rag_engine.query(query)
                
                # Afficher la réponse
                console.print("\n" + "─" * 100)
                console.print(Panel(result["answer"], title="Réponse", border_style="cyan"))
                
                # Afficher les sources
                if result["sources"]:
                    table = Table(show_header=True, header_style="bold")
                    table.add_column("Sources utilisées", style="dim")
                    for source in result["sources"]:
                        table.add_row(source)
                    console.print(table)
                
                console.print(f"[dim]Confiance: {result['confidence']:.2%}[/]")
            
            else:
                # Mode recherche simple
                with console.status("[cyan]Recherche en cours..."):
                    results = search_engine.search(query)
                
                if not results:
                    console.print("[yellow]Aucun résultat pertinent trouvé.[/]")
                    continue
                
                # Afficher les résultats
                table = Table(show_header=True, header_style="bold")
                table.add_column("Score", justify="right")
                table.add_column("Document", style="cyan")
                table.add_column("Extrait", style="green")
                
                for result in results:
                    table.add_row(
                        result["score"],
                        result["document"],
                        result["excerpt"]
                    )
                
                console.print(table)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Recherche terminée.[/]")
    except Exception as e:
        console.print(f"[red]Erreur: {str(e)}[/]")

if __name__ == "__main__":
    main()