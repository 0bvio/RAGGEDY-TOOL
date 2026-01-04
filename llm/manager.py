import os
import json
import requests
from tqdm import tqdm
from typing import List, Dict, Optional, Callable
try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    HfApi = None
    hf_hub_download = None

class ModelManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.metadata_path = os.path.join(self.models_dir, "metadata.json")
        os.makedirs(self.models_dir, exist_ok=True)
        self.metadata = self._load_metadata()
        # Default models to suggest
        self.recommended_models = [
            {
                "name": "Llama-3-8B-Instruct-GGUF (Q4_K_M)",
                "repo": "bartowski/Meta-Llama-3-8B-Instruct-GGUF",
                "file": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
                "url": "https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
            },
            {
                "name": "Mistral-7B-Instruct-v0.3-GGUF (Q4_K_M)",
                "repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
                "file": "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
                "url": "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"
            }
        ]

    def _load_metadata(self) -> Dict:
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_metadata(self):
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def list_local_models(self) -> List[str]:
        return sorted([f for f in os.listdir(self.models_dir) if f.endswith(".gguf")])

    def get_nickname(self, filename: str) -> str:
        return self.metadata.get(filename, {}).get("nickname", filename)

    def set_nickname(self, filename: str, nickname: str):
        if filename not in self.metadata:
            self.metadata[filename] = {}
        self.metadata[filename]["nickname"] = nickname
        self._save_metadata()

    def get_task_model(self, task: str) -> Optional[str]:
        """Returns the filename of the model assigned to a specific task."""
        return self.metadata.get("task_assignments", {}).get(task)

    def set_task_model(self, task: str, filename: Optional[str]):
        """Assigns a model to a specific task."""
        if "task_assignments" not in self.metadata:
            self.metadata["task_assignments"] = {}
        if filename:
            self.metadata["task_assignments"][task] = filename
        elif task in self.metadata["task_assignments"]:
            del self.metadata["task_assignments"][task]
        self._save_metadata()

    def delete_model(self, filename: str):
        path = self.get_model_path(filename)
        if os.path.exists(path):
            os.remove(path)
        if filename in self.metadata:
            del self.metadata[filename]
            self._save_metadata()

    def search_huggingface(self, query: str) -> List[Dict]:
        if not HfApi:
            return []
        api = HfApi()
        # We search for GGUF models specifically
        models = api.list_models(
            search=query,
            filter="gguf",
            sort="downloads",
            direction=-1,
            limit=20
        )
        
        results = []
        for model in models:
            # We want to find actual .gguf files in the repo
            try:
                files = api.list_repo_files(repo_id=model.modelId)
                gguf_files = [f for f in files if f.endswith(".gguf")]
                if gguf_files:
                    results.append({
                        "id": model.modelId,
                        "author": model.author,
                        "downloads": getattr(model, "downloads", 0),
                        "files": gguf_files
                    })
            except:
                continue
        return results

    def download_model(self, model_url: str, filename: str, progress_callback: Optional[Callable[[int, int], None]] = None):
        target_path = os.path.join(self.models_dir, filename)
        if os.path.exists(target_path):
            return target_path

        # If it's a full URL, use requests
        if model_url.startswith("http"):
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))

            with open(target_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024*1024):
                    size = f.write(data)
                    bar.update(size)
                    if progress_callback:
                        progress_callback(bar.n, total_size)
        else:
            # Assume it's a repo/id and use hf_hub_download
            # But the UI currently passes URLs for recommended models.
            # We'll handle both.
            pass
        
        return target_path

    def download_from_hf(self, repo_id: str, filename: str, progress_callback: Optional[Callable[[int, int], None]] = None):
        if not hf_hub_download:
            return None
            
        # hf_hub_download handles progress via tqdm, but we want our own callback for Streamlit
        # We can wrap it or just use the URL approach if we get the URL.
        # Actually, hf_hub_download is cleaner for repo_id.
        
        target_path = os.path.join(self.models_dir, filename)
        
        # We use the same requests approach but we need the URL
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        return self.download_model(url, filename, progress_callback)

    def get_model_path(self, filename: str) -> str:
        return os.path.join(self.models_dir, filename)
