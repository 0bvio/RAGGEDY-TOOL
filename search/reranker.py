import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict
from utils.logger import raggedy_logger
from utils.resource_monitor import ResourceMonitor

class Reranker:
    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.available = False
        self.monitor = ResourceMonitor()
        raggedy_logger.info(f"Reranker initialized (lazy load): {model_name}")

    def _load_model(self):
        if self.model is not None:
            return True
        
        # Resource check: bge-reranker-v2-m3 is ~2.2GB
        # Be very conservative: if >80% VRAM or <3GB free, avoid CUDA
        stats = self.monitor.get_system_stats()
        if stats.get("vram_available"):
            if stats["vram_percent"] > 80 or stats["vram_free_gb"] < 3.0:
                raggedy_logger.warning(f"VRAM limited ({stats['vram_percent']}% used, {stats['vram_free_gb']}GB free). Attempting Reranker CPU fallback.")
                device = "cpu"
            else:
                device = "cuda"
        else:
            device = "cpu"

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(device)
            self.model.eval()
            self.available = True
            raggedy_logger.info(f"Loaded Reranker model: {self.model_name} on {device}")
            return True
        except Exception as e:
            raggedy_logger.error(f"Failed to load Reranker: {e}")
            self.available = False
            return False

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        if not chunks:
            return []
            
        if not self._load_model():
            return chunks[:top_k]

        pairs = [[query, chunk['text']] for chunk in chunks]
        device = next(self.model.parameters()).device
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            
        # Combine chunks with scores
        scored_chunks = []
        for i, score in enumerate(scores):
            chunk = chunks[i].copy()
            chunk['rerank_score'] = float(score)
            scored_chunks.append(chunk)
            
        # Sort by score
        scored_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        return scored_chunks[:top_k]
