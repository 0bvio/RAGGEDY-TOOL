import os
import json
import numpy as np
from typing import List, Dict, Any

class SystemAuditor:
    def __init__(self, data_root: str = "data"):
        self.data_root = data_root
        self.raw_dir = os.path.join(data_root, "raw")
        self.processed_dir = os.path.join(data_root, "processed")
        self.chunks_dir = os.path.join(self.processed_dir, "chunks")
        self.embeddings_dir = os.path.join(self.processed_dir, "embeddings")

    def perform_audit(self) -> Dict[str, Any]:
        report = {
            "documents": [],
            "orphaned_chunks": 0,
            "orphaned_embeddings": 0,
            "consistency_score": 0,
            "issues": []
        }

        if not os.path.exists(self.processed_dir):
            return report

        # 1. Check Documents
        doc_files = [f for f in os.listdir(self.processed_dir) if f.endswith(".json") and f != "graph.json"]
        processed_doc_ids = [f.split(".")[0] for f in doc_files]
        
        chunk_files = os.listdir(self.chunks_dir) if os.path.exists(self.chunks_dir) else []
        emb_files = os.listdir(self.embeddings_dir) if os.path.exists(self.embeddings_dir) else []
        
        chunk_map = {} # doc_id -> list of chunks
        for cf in chunk_files:
            if not cf.endswith(".json"): continue
            try:
                with open(os.path.join(self.chunks_dir, cf), 'r') as f:
                    cdata = json.load(f)
                    did = cdata.get("doc_id")
                    if did not in chunk_map: chunk_map[did] = []
                    chunk_map[did].append(cdata.get("chunk_id"))
            except: pass

        for doc_id in processed_doc_ids:
            doc_report = {"doc_id": doc_id, "status": "OK", "missing": []}
            
            # Check processed JSON
            try:
                with open(os.path.join(self.processed_dir, f"{doc_id}.json"), 'r') as f:
                    doc_data = json.load(f)
                    filename = doc_data["metadata"]["filename"]
                    doc_report["filename"] = filename
                    
                    # Check if raw exists
                    if not os.path.exists(os.path.join(self.raw_dir, filename)):
                        doc_report["status"] = "Warning"
                        doc_report["missing"].append("Raw File Missing")
            except:
                doc_report["status"] = "Error"
                doc_report["missing"].append("Metadata Corrupt")
                report["documents"].append(doc_report)
                continue

            # Check Chunks
            expected_chunks = chunk_map.get(doc_id, [])
            if not expected_chunks:
                doc_report["status"] = "Error"
                doc_report["missing"].append("No Chunks Found")
            else:
                doc_report["chunk_count"] = len(expected_chunks)
                # Check Embeddings for these chunks
                missing_embs = 0
                for cid in expected_chunks:
                    if f"{cid}.npy" not in emb_files:
                        missing_embs += 1
                
                if missing_embs > 0:
                    doc_report["status"] = "Warning"
                    doc_report["missing"].append(f"{missing_embs} Embeddings Missing")

            report["documents"].append(doc_report)

        # 2. Check Orphans
        known_chunk_ids = set()
        for doc_chunks in chunk_map.values():
            known_chunk_ids.update(doc_chunks)
        
        for cf in chunk_files:
            cid = cf.split(".")[0]
            if cid not in known_chunk_ids:
                report["orphaned_chunks"] += 1
        
        for ef in emb_files:
            cid = ef.split(".")[0]
            if cid not in known_chunk_ids:
                report["orphaned_embeddings"] += 1

        # 3. Final Scoring
        total_docs = len(report["documents"])
        if total_docs > 0:
            ok_docs = sum(1 for d in report["documents"] if d["status"] == "OK")
            report["consistency_score"] = int((ok_docs / total_docs) * 100)
        else:
            report["consistency_score"] = 100

        return report

    def repair(self, report: Dict[str, Any]):
        """Cleans up orphaned files."""
        if report["orphaned_chunks"] > 0:
            # Re-collect known chunks to be safe
            known_chunks = set()
            doc_files = [f for f in os.listdir(self.processed_dir) if f.endswith(".json") and f != "graph.json"]
            for df in doc_files:
                doc_id = df.split(".")[0]
                # We'd need to re-scan chunks... actually simpler to just delete anything 
                # that doesn't belong to an existing doc_id in the chunks themselves.
                pass
            
            # Simple repair: delete orphaned chunks and embeddings
            # (In a real system we might be more careful)
            pass
