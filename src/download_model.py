import os
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from rich.console import Console

console = Console()

def download_llama_model():
    # Charger les variables d'environnement
    load_dotenv()
    
    # Vérifier le token
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        console.print("[red]Erreur: HUGGINGFACE_TOKEN non trouvé dans .env[/]")
        console.print("Veuillez créer un fichier .env avec votre token Hugging Face:")
        console.print("HUGGINGFACE_TOKEN=your_token_here")
        return False

    model_path = "models/llama-2-3b-chat.gguf"
    if not os.path.exists("models"):
        os.makedirs("models")
        
    if not os.path.exists(model_path):
        try:
            console.print("[cyan]Téléchargement du modèle Llama 2...[/]")
            hf_hub_download(
                repo_id="TheBloke/Llama-2-7B-Chat-GGUF",  # Modèle mis à jour
                filename="llama-2-7b-chat.Q4_K_M.gguf",    # Nouveau nom de fichier
                local_dir="models",
                token=token
            )
            os.rename(
                "models/llama-2-7b-chat.Q4_K_M.gguf",
                model_path
            )
            console.print("[green]Modèle téléchargé avec succès![/]")
            return True
        except Exception as e:
            console.print(f"[red]Erreur lors du téléchargement: {str(e)}[/]")
            console.print("[yellow]Message: Assurez-vous d'avoir accepté les conditions d'utilisation de Llama 2 sur Hugging Face[/]")
            console.print("[yellow]Visitez: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf[/]")
            return False
    else:
        console.print("[green]Le modèle existe déjà![/]")
        return True

if __name__ == "__main__":
    download_llama_model()