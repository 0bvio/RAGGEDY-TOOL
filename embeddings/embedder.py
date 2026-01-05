import os
import json
import threading
import numpy as np
from typing import List, Dict, Optional
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from qdrant_client import QdrantClient
from qdrant_client.http import models
from utils.logger import raggedy_logger
from utils.resource_monitor import ResourceMonitor

class Embedder:
    def __init__(self, model_name: str = 'BAAI/bge-large-en-v1.5', gpu_lock: Optional[threading.Lock] = None, qdrant_host: str = "localhost:6333"):
        self.model_name = model_name
        self.model = None
        self.monitor = ResourceMonitor()
        self.gpu_lock = gpu_lock
        
        # Qdrant setup
        self.qdrant_host = os.getenv("QDRANT_HOST", qdrant_host)
        self.qdrant = QdrantClient(host=self.qdrant_host.split(":")[0], port=int(self.qdrant_host.split(":")[1]))
        self.collection_name = "raggedy_embeddings"
        self._create_collection_if_not_exists()
        
        raggedy_logger.info(f"Embedder initialized (lazy load): {model_name}, Qdrant: {self.qdrant_host}")

    def _create_collection_if_not_exists(self):
        try:
            exists = self.qdrant.collection_exists(self.collection_name)
            if not exists:
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
                )
                raggedy_logger.info(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            raggedy_logger.warning(f"Qdrant not available: {e}. Falling back to local storage.")

    def upload_embeddings(self, chunks: List[Dict], embeddings: List[np.ndarray]):
        """Upload embeddings to Qdrant"""
        try:
            exists = self.qdrant.collection_exists(self.collection_name)
        except Exception:
            exists = False
        if not exists:
            raggedy_logger.warning("Qdrant collection not available, skipping upload")
            return
        
        points = []
        for chunk, emb in zip(chunks, embeddings):
            point = models.PointStruct(
                id=chunk["id"],
                vector=emb.tolist(),
                payload={
                    "text": chunk["text"],
                    "chunk_id": chunk["id"],
                    "doc_id": chunk.get("doc_id", "")
                }
            )
            points.append(point)
        
        self.qdrant.upsert(collection_name=self.collection_name, points=points)
        raggedy_logger.info(f"Uploaded {len(points)} embeddings to Qdrant")

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
            import contextlib
            
            # Use lock if available
            lock_ctx = self.gpu_lock if self.gpu_lock and device == "cuda" else contextlib.nullcontext()
            
            with lock_ctx:
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
        
        import contextlib
        device = getattr(self.model, "device", "cpu")
        lock_ctx = self.gpu_lock if self.gpu_lock and "cuda" in str(device) else contextlib.nullcontext()
        
        with lock_ctx:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return [np.array(e) for e in embeddings]

    def save_embeddings(self, chunks: List[Dict], embeddings: List[np.ndarray], output_dir: str):
        if not embeddings:
            return
        os.makedirs(output_dir, exist_ok=True)
        for chunk, embedding in zip(chunks, embeddings):
            emb_path = os.path.join(output_dir, f"{chunk['chunk_id']}.npy")
            np.save(emb_path, embedding)
