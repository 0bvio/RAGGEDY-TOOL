import os
import json
from typing import List, Dict

class DataPoolManager:
    def __init__(self, storage_dir: str = "data"):
        self.storage_path = os.path.join(storage_dir, "pools.json")
        self.pools = self._load_pools()

    def _load_pools(self) -> Dict[str, List[str]]:
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_pools(self):
        with open(self.storage_path, 'w') as f:
            json.dump(self.pools, f, indent=2)

    def create_pool(self, name: str, doc_ids: List[str] = None):
        self.pools[name] = doc_ids or []
        self._save_pools()

    def delete_pool(self, name: str):
        if name in self.pools:
            del self.pools[name]
            self._save_pools()

    def add_to_pool(self, name: str, doc_id: str):
        if name not in self.pools:
            self.pools[name] = []
        if doc_id not in self.pools[name]:
            self.pools[name].append(doc_id)
            self._save_pools()

    def remove_from_pool(self, name: str, doc_id: str):
        if name in self.pools and doc_id in self.pools[name]:
            self.pools[name].remove(doc_id)
            self._save_pools()

    def get_pool_docs(self, name: str) -> List[str]:
        return self.pools.get(name, [])

    def list_pools(self) -> List[str]:
        return sorted(list(self.pools.keys()))

    def remove_doc_from_all_pools(self, doc_id: str):
        updated = False
        for pool_name in self.pools:
            if doc_id in self.pools[pool_name]:
                self.pools[pool_name].remove(doc_id)
                updated = True
        if updated:
            self._save_pools()
