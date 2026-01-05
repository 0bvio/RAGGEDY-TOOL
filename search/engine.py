import os
import json
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from utils.logger import raggedy_logger

class SearchEngine:
    def __init__(self, chunk_dir: str, embedding_dir: str, es_host: str = "localhost:9200", qdrant_host: str = "localhost:6333"):
        self.chunk_dir = chunk_dir
        self.embedding_dir = embedding_dir
        self.chunks = []
        self.embeddings = []
        self.chunk_ids = []
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        
        # Elasticsearch setup
        self.es_host = os.getenv("ELASTICSEARCH_HOST", es_host)
        self.es = Elasticsearch([f"http://{self.es_host}"])
        self.index_name = "raggedy_chunks"
        self._create_index_if_not_exists()
        
        # Qdrant setup
        self.qdrant_host = os.getenv("QDRANT_HOST", qdrant_host)
        self.qdrant = QdrantClient(host=self.qdrant_host.split(":")[0], port=int(self.qdrant_host.split(":")[1]))
        self.collection_name = "raggedy_embeddings"
        
        self._load_data()

    def _create_index_if_not_exists(self):
        try:
            if not self.es.indices.exists(index=self.index_name):
                mapping = {
                    "mappings": {
                        "properties": {
                            "text": {"type": "text", "analyzer": "standard"},
                            "chunk_id": {"type": "keyword"},
                            "doc_id": {"type": "keyword"}
                        }
                    }
                }
                self.es.indices.create(index=self.index_name, body=mapping)
                raggedy_logger.info(f"Created Elasticsearch index: {self.index_name}")
        except Exception as e:
            raggedy_logger.warning(f"Elasticsearch not available: {e}. Falling back to local search.")

    def index_chunks(self, chunks: List[Dict]):
        """Index chunks in Elasticsearch"""
        if not self.es.ping():
            raggedy_logger.warning("Elasticsearch not available, skipping indexing")
            return
        
        for chunk in chunks:
            doc = {
                "text": chunk["text"],
                "chunk_id": chunk["id"],
                "doc_id": chunk.get("doc_id", "")
            }
            self.es.index(index=self.index_name, id=chunk["id"], body=doc)
        raggedy_logger.info(f"Indexed {len(chunks)} chunks in Elasticsearch")

    def _load_data(self):
        if not os.path.exists(self.chunk_dir):
            return
            
        for filename in os.listdir(self.chunk_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.chunk_dir, filename), 'r') as f:
                    chunk = json.load(f)
                    self.chunks.append(chunk)
                    self.chunk_ids.append(chunk['chunk_id'])
                    
                    emb_path = os.path.join(self.embedding_dir, f"{chunk['chunk_id']}.npy")
                    if os.path.exists(emb_path):
                        self.embeddings.append(np.load(emb_path))
                    else:
                        # Placeholder if no embedding found
                        self.embeddings.append(np.zeros(1024))
        
        if self.chunks:
            texts = [c['text'] for c in self.chunks]
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Filter out embeddings with wrong dimensions
            valid_embeddings = []
            valid_chunks = []
            valid_chunk_ids = []
            for i, emb in enumerate(self.embeddings):
                if emb.shape == (1024,):
                    valid_embeddings.append(emb)
                    valid_chunks.append(self.chunks[i])
                    valid_chunk_ids.append(self.chunk_ids[i])
                else:
                    # If dimension is wrong (e.g. old 384-dim), we skip it for vector search
                    # but it remains in chunks for lexical if we were careful.
                    # Actually, better to just keep them all and use a zero vector of correct size
                    valid_embeddings.append(np.zeros(1024))
                    valid_chunks.append(self.chunks[i])
                    valid_chunk_ids.append(self.chunk_ids[i])
            
            self.embeddings = np.array(valid_embeddings)
            self.chunks = valid_chunks
            self.chunk_ids = valid_chunk_ids

    def search_lexical(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        if not self.es.ping():
            # Fallback to local TF-IDF
            if self.tfidf_matrix is None:
                return []
            query_vec = self.tfidf_vectorizer.transform([query])
            scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
            top_indices = np.argsort(scores)[-top_k:][::-1]
            return [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]
        
        body = {
            "query": {
                "match": {
                    "text": query
                }
            },
            "size": top_k
        }
        response = self.es.search(index=self.index_name, body=body)
        results = []
        for hit in response["hits"]["hits"]:
            chunk_data = {
                "id": hit["_source"]["chunk_id"],
                "text": hit["_source"]["text"],
                "doc_id": hit["_source"]["doc_id"]
            }
            score = hit["_score"]
            results.append((chunk_data, score))
        return results

    def search_vector(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        try:
            exists = self.qdrant.collection_exists(self.collection_name)
        except Exception:
            exists = False
        if not exists:
            # Fallback to local
            return self._search_vector_local(query_emb, top_k)
            
            # Ensure query_emb is 1024
            if query_emb.shape != (1024,):
                if query_emb.shape[0] < 1024:
                    query_emb = np.pad(query_emb, (0, 1024 - query_emb.shape[0]))
                else:
                    query_emb = query_emb[:1024]
            
            search_result = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_emb.tolist(),
                limit=top_k
            )
            results = []
            for hit in search_result:
                chunk_data = {
                    "id": hit.payload["chunk_id"],
                    "text": hit.payload["text"],
                    "doc_id": hit.payload["doc_id"]
                }
                score = hit.score
                results.append((chunk_data, score))
            return results
        except Exception as e:
            raggedy_logger.warning(f"Qdrant search failed: {e}. Falling back to local.")
            return self._search_vector_local(query_emb, top_k)

    def _search_vector_local(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Fallback local vector search"""
        if len(self.embeddings) == 0:
            return []
        
        # Filter out chunks that have zero vectors (placeholders)
        valid_mask = np.any(self.embeddings != 0, axis=1)
        if not np.any(valid_mask):
            return []
            
        valid_embs = self.embeddings[valid_mask]
        valid_chunks = [c for i, c in enumerate(self.chunks) if valid_mask[i]]

        # Ensure query_emb is also 1024
        if query_emb.shape != (1024,):
            if query_emb.shape[0] < 1024:
                query_emb = np.pad(query_emb, (0, 1024 - query_emb.shape[0]))
            else:
                query_emb = query_emb[:1024]

        # Cosine similarity
        norm_query = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        norm_embs = valid_embs / (np.linalg.norm(valid_embs, axis=1, keepdims=True) + 1e-9)
        scores = np.dot(norm_embs, norm_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(valid_chunks[i], float(scores[i])) for i in top_indices]

    def search_hybrid(self, query: str, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float, str]]:
        lexical_results = {c['chunk_id']: score for c, score in self.search_lexical(query, top_k*4)}
        vector_results = {c['chunk_id']: score for c, score in self.search_vector(query_emb, top_k*4)}
        
        all_ids = set(lexical_results.keys()) | set(vector_results.keys())
        combined = []
        for cid in all_ids:
            lex_score = lexical_results.get(cid, 0)
            vec_score = vector_results.get(cid, 0)
            
            # Use a slightly more sophisticated score normalization/fusion for real models
            # Here we just keep the sum of scores for candidates
            score = lex_score + vec_score
            
            method = []
            if lex_score > 0: method.append("lexical")
            if vec_score > 0: method.append("vector")
            
            try:
                chunk = next(c for c in self.chunks if c['chunk_id'] == cid)
                combined.append((chunk, score, " + ".join(method)))
            except StopIteration:
                continue
            
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]
