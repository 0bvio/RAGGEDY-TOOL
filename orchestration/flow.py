import os
import json
import numpy as np
from typing import List, Dict, Optional
from ingestion.ingestor import Ingestor
from ingestion.pool_manager import DataPoolManager
from chunking.chunker import Chunker
from embeddings.embedder import Embedder
from search.engine import SearchEngine
from search.reranker import Reranker
from graph.graph_store import GraphStore
from llm.client import LLMClient
from utils.logger import raggedy_logger

class RaggedyOrchestrator:
    def __init__(self, data_root: str = "data"):
        raggedy_logger.info(f"Initializing RaggedyOrchestrator with root: {data_root}")
        self.raw_dir = os.path.join(data_root, "raw")
        self.processed_dir = os.path.join(data_root, "processed")
        self.chunks_dir = os.path.join(self.processed_dir, "chunks")
        self.embeddings_dir = os.path.join(self.processed_dir, "embeddings")
        self.graph_path = os.path.join(self.processed_dir, "graph.json")
        
        self.ingestor = Ingestor(self.raw_dir, self.processed_dir)
        self.pool_manager = DataPoolManager(data_root)
        self.chunker = Chunker()
        self.graph_store = GraphStore(self.graph_path)
        self.llm = LLMClient()
        self.embedder = Embedder(gpu_lock=self.llm.gpu_lock)
        self.reranker = Reranker()
        os.makedirs(self.chunks_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        self.search_engine = None

    def process_new_documents(self, specific_docs: List[Dict] = None):
        raggedy_logger.info("Starting ingestion processing...")
        # 1. Ingest (if not specific_docs provided)
        if specific_docs is None:
            docs = self.ingestor.ingest_all()
        else:
            docs = specific_docs
            
        all_chunks = []
        
        for doc in docs:
            doc_id = doc['metadata']['doc_id']
            filename = doc['metadata']['filename']
            raggedy_logger.info(f"Processing document: {filename} ({doc_id})")
            
            # Fetch full doc to update logs
            full_doc = self.ingestor.get_document_data(doc_id)
            if not full_doc:
                full_doc = doc
            
            if "ingestion_logs" not in full_doc:
                full_doc["ingestion_logs"] = []
            
            # 2. Chunk
            raggedy_logger.info(f"  - Chunking text into small units for processing...")
            chunks = self.chunker.chunk_document(full_doc)
            self.chunker.save_chunks(chunks, self.chunks_dir)
            all_chunks.extend(chunks)
            raggedy_logger.info(f"  - Created {len(chunks)} chunks")
            full_doc["ingestion_logs"].append(f"Created {len(chunks)} chunks")
            
            # 3. Embed
            raggedy_logger.info(f"  - Generating vector embeddings for semantic search...")
            embeddings = self.embedder.embed_chunks(chunks)
            if embeddings:
                self.embedder.save_embeddings(chunks, embeddings, self.embeddings_dir)
                self.embedder.upload_embeddings(chunks, embeddings)
                raggedy_logger.info(f"  - Generated {len(embeddings)} embeddings")
                full_doc["ingestion_logs"].append(f"Generated {len(embeddings)} vector embeddings using {self.embedder.model_name if self.embedder.model else 'fallback'}")
            else:
                raggedy_logger.warning(f"  - FAILED to generate embeddings for {filename}. Check logs.")
                full_doc["ingestion_logs"].append("FAILED to generate vector embeddings")
            
            # 4. Graph Extraction (Layer 4)
            raggedy_logger.info(f"  - Extracting conceptual relationships for Knowledge Graph...")
            graph_count = 0
            if self.llm.is_available():
                for i, chunk in enumerate(chunks):
                    if i % 5 == 0: # Log progress for large docs
                        raggedy_logger.info(f"    - Processing graph extraction for chunk {i+1}/{len(chunks)}...")
                    entities_rels = self.llm.extract_entities_relations(chunk['text'])
                    for item in entities_rels:
                        self.graph_store.add_relationship(
                            item['source'], 
                            item['target'], 
                            item['relation'], 
                            item.get('confidence', 0.5),
                            chunk['chunk_id']
                        )
                        graph_count += 1
                raggedy_logger.info(f"  - Extracted {graph_count} graph relationships")
                full_doc["ingestion_logs"].append(f"Extracted {graph_count} conceptual relationships to graph")
            else:
                raggedy_logger.warning("  - LLM server offline. Skipping graph extraction.")
                full_doc["ingestion_logs"].append("SKIPPED graph extraction (LLM Offline)")
            
            # Save updated doc with new logs
            output_path = os.path.join(self.processed_dir, f"{doc_id}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(full_doc, f, indent=2)
        
        self.graph_store.save()
        # Refresh search engine
        self.search_engine = SearchEngine(self.chunks_dir, self.embeddings_dir)
        # Index chunks in Elasticsearch
        self.search_engine.index_chunks(all_chunks)

    def delete_document(self, doc_id: str):
        raggedy_logger.info(f"Deleting document and associated data: {doc_id}")
        
        # 1. Delete metadata/content
        self.ingestor.delete_document(doc_id)
        
        # 2. Delete chunks
        chunks_to_remove = []
        if os.path.exists(self.chunks_dir):
            for filename in os.listdir(self.chunks_dir):
                if filename.endswith(".json"):
                    path = os.path.join(self.chunks_dir, filename)
                    try:
                        with open(path, 'r') as f:
                            chunk = json.load(f)
                            if chunk.get('doc_id') == doc_id:
                                chunks_to_remove.append(chunk['chunk_id'])
                                os.remove(path)
                                # 3. Delete corresponding embedding
                                emb_path = os.path.join(self.embeddings_dir, f"{chunk['chunk_id']}.npy")
                                if os.path.exists(emb_path):
                                    os.remove(emb_path)
                    except:
                        continue
        
        # 4. Remove from graph
        self.graph_store.remove_by_chunk_ids(chunks_to_remove)
        self.graph_store.save()
        
        # 5. Remove from pools
        self.pool_manager.remove_doc_from_all_pools(doc_id)
        
        # 6. Refresh search engine
        self.search_engine = SearchEngine(self.chunks_dir, self.embeddings_dir)

    def ask(self, query: str, history: List[Dict] = None, doc_ids: List[str] = None, stream: bool = False, chat_id: Optional[str] = None, **kwargs) -> Dict:
        raggedy_logger.info(f"Query received: {query}")
        
        # Retrieval parameters from kwargs or defaults
        top_k_search = kwargs.get('top_k_search', 20)
        top_k_rerank = kwargs.get('top_k_rerank', 7)
        system_prompt = kwargs.get('system_prompt', None)
        
        # Check for pre-pooled data (AI Reflection)
        pre_pooled = kwargs.get('use_pooled_data')

        # Decide if we need retrieval
        needs_retrieval = True
        if doc_ids is not None and len(doc_ids) == 0:
            needs_retrieval = False
            raggedy_logger.info("No documents selected. Skipping Stage A retrieval.")
        
        # Also skip if no chunks exist at all
        if needs_retrieval and (not os.path.exists(self.chunks_dir) or not os.listdir(self.chunks_dir)):
            needs_retrieval = False
            raggedy_logger.info("Knowledge base is empty. Skipping Stage A retrieval.")

        query_emb = None
        chunk_methods = {}
        all_candidate_chunks_map = {}

        if pre_pooled:
            raggedy_logger.info("Using pre-pooled background data for AI reflection.")
            for chunk in pre_pooled.get("chunks", []):
                all_candidate_chunks_map[chunk['chunk_id']] = chunk
            chunk_methods = pre_pooled.get("methods", {})
        elif needs_retrieval:
            if not self.search_engine:
                self.search_engine = SearchEngine(self.chunks_dir, self.embeddings_dir)
            
            # 1. Get query embedding (Lazy load Embedder only if needed)
            raggedy_logger.info("Gathering query embedding for hybrid search...")
            query_embs = self.embedder.embed_chunks([{"text": query}])
            if query_embs:
                query_emb = query_embs[0]
            
                # Stage A: Initial Hybrid Retrieval (Layers 2 & 3)
                raggedy_logger.info(f"Executing Stage A hybrid search (top_k={top_k_search})...")
                retrieved = self.search_engine.search_hybrid(query, query_emb, top_k=top_k_search)
                if doc_ids is not None:
                    retrieved = [r for r in retrieved if r[0]['doc_id'] in doc_ids]

                for c, score, method in retrieved:
                    all_candidate_chunks_map[c['chunk_id']] = c
                    chunk_methods[c['chunk_id']] = method
        else:
            raggedy_logger.info("Skipping retrieval: No documents selected or knowledge base empty.")

        # 3. Quick Rerank for Initial Context (Lazy load Reranker only if candidates exist)
        initial_candidates = list(all_candidate_chunks_map.values())
        if initial_candidates:
            final_context_chunks = self.reranker.rerank(query, initial_candidates, top_k=top_k_rerank)
            # Filter by relevance threshold to avoid garbage context triggering RAG mode
            # BGE-Reranker-v2-m3 threshold: -5.0 is a safe bet for 'irrelevant'
            final_context_chunks = [c for c in final_context_chunks if c.get('rerank_score', 0) > -5.0]
        else:
            final_context_chunks = []
        
        # 4. Answer Generation (Layer 9)
        llm_params = {k: v for k, v in kwargs.items() if k not in ['top_k_search', 'top_k_rerank', 'system_prompt']}
        answer = self.llm.generate_answer(query, final_context_chunks, history=history, system_prompt=system_prompt, stream=stream, chat_id=chat_id, **llm_params)
        
        return {
            "query": query,
            "answer": answer,
            "sources": [{"index": i+1, "chunk_id": c['chunk_id'], "filename": c['filename'], "text": c['text'], "method": chunk_methods.get(c['chunk_id'], "unknown")} 
                        for i, c in enumerate(final_context_chunks)],
            "query_emb": query_emb, # Return for background use
            "all_candidate_chunks_map": all_candidate_chunks_map,
            "chunk_methods": chunk_methods
        }

    def pool_additional_data(self, query: str, history: List[Dict], query_emb: np.ndarray, 
                             existing_chunks_map: Dict, existing_methods: Dict, 
                             doc_ids: List[str] = None, chat_id: Optional[str] = None) -> Dict:
        """Background task to perform Stage B (Graph) and Stage C (Pooling)."""
        raggedy_logger.info(f"Background pooling for: {query}")
        
        chunk_methods = existing_methods.copy()
        all_candidate_chunks_map = existing_chunks_map.copy()
        trace = {
            "initial_query": query,
            "entities": [],
            "graph_expansions": [],
            "idea_pools": []
        }
        
        if doc_ids is not None and len(doc_ids) == 0:
            return {"chunks": [], "methods": {}, "trace": trace}

        # Stage B: Concept/Entity Expansion (Pooling from Graph)
        try:
            query_entities = [e['source'] for e in self.llm.extract_entities_relations(query, chat_id=chat_id)]
            trace["entities"] = query_entities
            
            if history:
                last_user = next((m for m in reversed(history) if m['role'] == 'user'), None)
                if last_user:
                    hist_content = ""
                    if "versions" in last_user:
                        hist_content = last_user["versions"][last_user.get("active_version", 0)]["content"]
                    else:
                        hist_content = last_user.get("content", "")
                    h_entities = [e['source'] for e in self.llm.extract_entities_relations(hist_content, chat_id=chat_id)]
                    query_entities.extend(h_entities)
                    trace["entities"] = list(set(trace["entities"] + h_entities))

            # Expand context via graph
            unique_entities = list(set(query_entities))
            expanded_chunk_ids = self.graph_store.expand_context(unique_entities, max_hops=1)
            
            for cid in expanded_chunk_ids:
                if cid in all_candidate_chunks_map:
                    if "graph" not in chunk_methods[cid]:
                        chunk_methods[cid] = chunk_methods[cid] + " + graph"
                    continue
                    
                chunk_path = os.path.join(self.chunks_dir, f"{cid}.json")
                if os.path.exists(chunk_path):
                    with open(chunk_path, 'r') as f:
                        c = json.load(f)
                        if doc_ids is not None and c['doc_id'] not in doc_ids:
                            continue
                        all_candidate_chunks_map[cid] = c
                        chunk_methods[cid] = "graph"
                        trace["graph_expansions"].append({"chunk_id": cid, "filename": c['filename']})

            # Stage C: Dynamic "Idea Pooling" (Search for extracted entities)
            for entity in unique_entities[:3]:
                entity_emb = self.embedder.embed_chunks([{"text": entity}])[0]
                entity_results = self.search_engine.search_vector(entity_emb, top_k=5)
                found_for_entity = 0
                for c, score in entity_results:
                    if doc_ids is not None and c['doc_id'] not in doc_ids:
                        continue
                    cid = c['chunk_id']
                    if cid not in all_candidate_chunks_map:
                        all_candidate_chunks_map[cid] = c
                        chunk_methods[cid] = f"pooling ({entity})"
                        found_for_entity += 1
                    else:
                        if f"pooling ({entity})" not in chunk_methods[cid]:
                            chunk_methods[cid] += f" + pooling ({entity})"
                
                if found_for_entity > 0:
                    trace["idea_pools"].append({"term": entity, "count": found_for_entity})
                    
        except Exception as e:
            raggedy_logger.error(f"Error in background pooling: {e}")

        return {
            "chunks": list(all_candidate_chunks_map.values()),
            "methods": chunk_methods,
            "trace": trace
        }

    def evaluate_proactive_thought(self, query: str, history: List[Dict], 
                                   initial_chunks: List[Dict], 
                                   pooled_chunks: List[Dict],
                                   chat_id: Optional[str] = None) -> Optional[str]:
        """Asks the LLM if the new pooled data offers significant new insights."""
        if not self.llm.is_available() or not pooled_chunks:
            return None
            
        # Filter for truly new chunks
        initial_ids = {c['chunk_id'] for c in initial_chunks}
        new_chunks = [c for c in pooled_chunks if c['chunk_id'] not in initial_ids]
        
        if not new_chunks:
            return None
            
        raggedy_logger.info(f"Evaluating {len(new_chunks)} new chunks for proactive thought...")
        
        # Limit to top new chunks for evaluation
        new_context = "\n\n".join([f"New Data Point:\n{c['text']}" for c in new_chunks[:5]])
        
        prompt = f"""Task: Analyze the new information found in the background and decide if it adds a significant new perspective or detail that wasn't covered in the initial answer to the user's query.

Initial Query: {query}

New Findings:
{new_context}

Rules:
1. If the new information is highly relevant and adds value, provide a brief (1-2 sentence) "Proactive Thought" or "Deep Dive" suggestion.
2. If it's redundant or irrelevant, return exactly "NONE".
3. Do NOT repeat the initial answer. Focus only on what is NEW.

Proactive Thought (or NONE):"""

        response = self.llm.complete(prompt, timeout=25, chat_id=chat_id)
        if "NONE" in response.upper() or len(response) < 10:
            return None
            
        return response.strip()
