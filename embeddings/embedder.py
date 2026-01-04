import os
import json
import numpy as np
from typing import List, Dict, Optional
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from utils.logger import raggedy_logger
from utils.resource_monitor import ResourceMonitor

class Embedder:
    def __init__(self, model_name: str = 'BAAI/bge-large-en-v1.5'):
        self.model_name = model_name
        self.model = None
        self.monitor = ResourceMonitor()
        raggedy_logger.info(f"Embedder initialized (lazy load): {model_name}")

    def _load_model(self):
        if self.model is not None:
            return True
        
        if not SentenceTransformer:
            raggedy_logger.error("sentence-transformers not installed.")
            return False

        # Resource check: bge-large is ~1.3GB
        # Be very conservative: if >85% VRAM or <2GB free, avoid CUDA
        stats = self.monitor.get_system_stats()
        if stats.get("vram_available"):
            if stats["vram_percent"] > 85 or stats["vram_free_gb"] < 2.0:
                raggedy_logger.warning(f"VRAM limited ({stats['vram_percent']}% used, {stats['vram_free_gb']}GB free). Attempting Embedder CPU fallback.")
                device = "cpu"
            else:
                device = "cuda"
        else:
            device = "cpu"

        try:
            import torch
            self.model = SentenceTransformer(self.model_name, device=device)
            raggedy_logger.info(f"Loaded Embedder model: {self.model_name} on {device}")
            return True
        except Exception as e:
            raggedy_logger.error(f"Failed to load Embedder: {e}")
            return False

    def embed_chunks(self, chunks: List[Dict]) -> List[np.ndarray]:
        if not chunks:
            return []
        
        if not self._load_model():
            return []

        texts = [c['text'] for c in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return [np.array(e) for e in embeddings]

    def save_embeddings(self, chunks: List[Dict], embeddings: List[np.ndarray], output_dir: str):
        if not embeddings:
            return
        os.makedirs(output_dir, exist_ok=True)
        for chunk, embedding in zip(chunks, embeddings):
            emb_path = os.path.join(output_dir, f"{chunk['chunk_id']}.npy")
            np.save(emb_path, embedding)
