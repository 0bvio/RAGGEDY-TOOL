import os
import json
import networkx as nx
from typing import List, Dict
from neo4j import GraphDatabase
from utils.logger import raggedy_logger

class GraphStore:
    def __init__(self, storage_path: str, neo4j_uri: str = "bolt://localhost:7687", neo4j_user: str = "neo4j", neo4j_password: str = "password"):
        self.storage_path = storage_path
        self.graph = nx.MultiDiGraph()
        
        # Neo4j setup
        self.neo4j_uri = os.getenv("NEO4J_URI", neo4j_uri)
        self.neo4j_user = os.getenv("NEO4J_USER", neo4j_user)
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", neo4j_password)
        try:
            self.driver = GraphDatabase.driver(self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password))
            self.neo4j_available = True
            raggedy_logger.info("Connected to Neo4j")
        except Exception as e:
            raggedy_logger.warning(f"Neo4j not available: {e}. Using local NetworkX.")
            self.neo4j_available = False
        
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
        # Add to Neo4j if available
        if self.neo4j_available:
            with self.driver.session() as session:
                session.run("""
                    MERGE (s:Entity {name: $source})
                    MERGE (t:Entity {name: $target})
                    MERGE (s)-[r:RELATION {type: $relation, evidence: $evidence}]->(t)
                    SET r.confidence = $confidence
                """, source=source, target=target, relation=relation, confidence=confidence, evidence=evidence_chunk_id)
        
        # Always update local graph
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
        if self.neo4j_available:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e:Entity {name: $entity})-[r:RELATION]-(other:Entity)
                    RETURN other.name, r.type, r.confidence, r.evidence, type(r) as direction
                """, entity=entity)
                neighbors = []
                for record in result:
                    neighbors.append({
                        "source": entity if record["direction"] == "OUTGOING" else record["other.name"],
                        "target": record["other.name"] if record["direction"] == "OUTGOING" else entity,
                        "relation": record["r.type"],
                        "confidence": record["r.confidence"],
                        "evidence": record["r.evidence"]
                    })
                return neighbors
        else:
            # Fallback to NetworkX
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
        if self.neo4j_available:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (e:Entity)-[*1..{hops}]-(connected:Entity)
                    WHERE e.name IN $entities
                    MATCH (connected)-[r:RELATION]-()
                    RETURN DISTINCT r.evidence
                """, entities=entities, hops=max_hops)
                return [record["r.evidence"] for record in result]
        else:
            # Fallback to NetworkX
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
