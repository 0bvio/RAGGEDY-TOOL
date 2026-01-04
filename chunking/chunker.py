import hashlib
import json
import os
from typing import List, Dict

class Chunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc_data: Dict) -> List[Dict]:
        content = doc_data["content"]
        doc_id = doc_data["metadata"]["doc_id"]
        filename = doc_data["metadata"]["filename"]
        
        chunks = []
        start = 0
        while start < len(content):
            end = start + self.chunk_size
            text = content[start:end]
            
            # Deterministic ID for the chunk
            chunk_hash = hashlib.sha256(f"{doc_id}:{start}:{text}".encode()).hexdigest()
            
            chunks.append({
                "chunk_id": chunk_hash,
                "doc_id": doc_id,
                "filename": filename,
                "text": text,
                "start_char": start,
                "end_char": end
            })
            
            start += (self.chunk_size - self.chunk_overlap)
            if start >= len(content) and len(chunks) > 0:
                break
                
        return chunks

    def save_chunks(self, chunks: List[Dict], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        for chunk in chunks:
            chunk_path = os.path.join(output_dir, f"{chunk['chunk_id']}.json")
            with open(chunk_path, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, indent=2)
