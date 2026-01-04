import os
import json
import networkx as nx
from typing import List, Dict

class GraphStore:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.graph = nx.MultiDiGraph()
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                # Handle both 'links' (old default) and 'edges' (new preferred) keys
                edge_key = "edges" if "edges" in data else "links"
                self.graph = nx.node_link_graph(data, edges=edge_key)

    def save(self):
        # We save using 'edges' for forward compatibility
        data = nx.node_link_data(self.graph, edges="edges")
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add_relationship(self, source: str, target: str, relation: str, confidence: float, evidence_chunk_id: str):
        if not self.graph.has_node(source):
            self.graph.add_node(source, type="entity")
        if not self.graph.has_node(target):
            self.graph.add_node(target, type="entity")
            
        self.graph.add_edge(
            source, 
            target, 
            relation=relation, 
            confidence=confidence, 
            evidence=evidence_chunk_id
        )

    def get_neighbors(self, entity: str) -> List[Dict]:
        if not self.graph.has_node(entity):
            return []
        
        neighbors = []
        for _, target, data in self.graph.out_edges(entity, data=True):
            neighbors.append({
                "source": entity,
                "target": target,
                "relation": data['relation'],
                "confidence": data['confidence'],
                "evidence": data['evidence']
            })
        return neighbors

    def remove_by_chunk_ids(self, chunk_ids: List[str]):
        """Removes all edges that link to any of the provided chunk IDs."""
        chunk_set = set(chunk_ids)
        to_remove = []
        for u, v, k, data in self.graph.edges(keys=True, data=True):
            if data.get('evidence') in chunk_set:
                to_remove.append((u, v, k))
        
        for u, v, k in to_remove:
            self.graph.remove_edge(u, v, key=k)
            
        # Optionally remove isolated nodes
        nodes_to_remove = [n for n in self.graph.nodes() if self.graph.degree(n) == 0]
        for n in nodes_to_remove:
            self.graph.remove_node(n)

    def expand_context(self, entities: List[str], max_hops: int = 1) -> List[str]:
        # Return evidence chunk IDs related to these entities
        evidence_ids = set()
        visited = set()
        to_visit = set(entities)
        
        for _ in range(max_hops):
            next_visit = set()
            for node in to_visit:
                if node in visited or not self.graph.has_node(node):
                    continue
                visited.add(node)
                for _, target, data in self.graph.out_edges(node, data=True):
                    evidence_ids.add(data['evidence'])
                    next_visit.add(target)
            to_visit = next_visit
            
        return list(evidence_ids)
