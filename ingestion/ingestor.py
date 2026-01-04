import hashlib
import json
import os
from datetime import datetime
from typing import List, Dict, Any
from ingestion.processors import TextProcessor, JsonProcessor, PdfProcessor, DocxProcessor, HtmlProcessor

class Ingestor:
    def __init__(self, raw_dir: str, processed_dir: str):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        self.processors = [
            TextProcessor(),
            JsonProcessor(),
            PdfProcessor(),
            DocxProcessor(),
            HtmlProcessor()
        ]

    def get_ingested_documents(self) -> List[Dict]:
        docs = []
        if not os.path.exists(self.processed_dir):
            return []
        for filename in os.listdir(self.processed_dir):
            if filename.endswith(".json") and filename != "graph.json":
                try:
                    with open(os.path.join(self.processed_dir, filename), 'r') as f:
                        data = json.load(f)
                        if "metadata" in data:
                            docs.append(data["metadata"])
                except:
                    continue
        return docs

    def ingest_file(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.raw_dir, filename)
        
        processor = next((p for p in self.processors if p.can_handle(filename)), TextProcessor())
        
        processed_data = processor.process(path)
        content = processed_data["content"]
        
        doc_id = hashlib.sha256(content.encode()).hexdigest()
        
        metadata = {
            "doc_id": doc_id,
            "filename": filename,
            "ingested_at": datetime.now().isoformat(),
            "content_hash": doc_id,
            "processor": processor.__class__.__name__,
            **processed_data.get("metadata", {})
        }
        
        output_path = os.path.join(self.processed_dir, f"{doc_id}.json")
        data = {
            "metadata": metadata,
            "content": content,
            "ingestion_logs": processed_data.get("logs", [])
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        return data

    def get_document_data(self, doc_id: str) -> Dict[str, Any]:
        path = os.path.join(self.processed_dir, f"{doc_id}.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def delete_document(self, doc_id: str):
        """Removes the document metadata and content from processed directory."""
        path = os.path.join(self.processed_dir, f"{doc_id}.json")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def ingest_all(self) -> List[Dict]:
        results = []
        ingested_ids = {doc['doc_id'] for doc in self.get_ingested_documents()}
        
        for filename in os.listdir(self.raw_dir):
            file_path = os.path.join(self.raw_dir, filename)
            if os.path.isfile(file_path):
                # We still ingest it to see if content changed, or we can check file hash first.
                # But for now, let's just ingest and if doc_id matches we might skip.
                # Actually, ingest_file returns the full data.
                result = self.ingest_file(filename)
                results.append(result)
        return results
