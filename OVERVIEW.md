# RAGGEDY TOOL Overview

## Project Overview
RAGGEDY TOOL is a minimally viable product (MVP) for an offline knowledge intelligence system designed to ingest unstructured text data, organize it into concepts and relationships, and provide grounded, citation-backed answers to natural language queries. It operates entirely offline, using local large language models (LLMs) and vector databases, without relying on cloud services or internet access. The system emphasizes "soft probabilistic graphs" for handling ambiguous or evolving knowledge, multi-stage retrieval (lexical + semantic + graph-based), and sentence-level citations for verifiability. It positions itself as a personal research engine and reasoning assistant, not a generic chatbot, focusing on complex, multi-step questions with evidence-backed responses.

The project is built in Python and follows a layered architecture inspired by Retrieval-Augmented Generation (RAG) principles, with additional components for knowledge graph construction, reranking, and resource management. It supports GPU acceleration via CUDA for performance but falls back to CPU. The codebase is structured for extensibility, with modular components for ingestion, chunking, embedding, search, graph storage, LLM interaction, and orchestration.

## Architecture
The system follows a multi-layered RAG pipeline, as outlined in the project design, with 15 technical development stages:
1. **System Foundation**: Docker-based runtime with persistent volumes, NVIDIA GPU passthrough, and services like llama.cpp (LLM inference), Elasticsearch (lexical search), Qdrant (vector DB), Neo4j (graph), and Python orchestration.
2. **Data Ingestion**: Verbatim text preservation with metadata tracking.
3. **Text Chunking & Normalization**: Deterministic splitting into overlapping chunks.
4. **Semantic Embeddings**: Vector representations using SentenceTransformers.
5. **Lexical & Hybrid Search**: BM25 + vector similarity fusion.
6. **Knowledge Graph Construction**: Probabilistic entity-relation extraction via LLM.
7. **Hybrid Retrieval & Graph Expansion**: Multi-hop context gathering.
8. **Reranking & Multi-Hop Reasoning**: Cross-encoder reranking and graph constraints.
9. **Answer Generation**: Grounded responses with structured prompts.
10. **Sentence-Level Citation**: Evidence mapping with validation.
11. **Self-Repair & Index Integrity**: Checksum-based rebuilds.
12. **Orchestration & Flow Control**: LangGraph-inspired state machine.
13. **Resource & Performance Management**: GPU memory awareness and job scheduling.
14. **User Interaction Layer**: CLI/API/UI interfaces.
15. **Final Acceptance**: Fully offline, citation-backed answers.

**Current Implementation Status**: MVP with Docker foundation established (docker-compose.yml, Dockerfile). Elasticsearch for lexical search, Qdrant for vector search, Neo4j for knowledge graph (with local NetworkX fallbacks). Full LangGraph orchestration and advanced features (graph viz, self-repair) pending.

## Main Components
- **Entry Points**:
  - `main.py`: CLI with commands (`ingest`, `query`, `ui`).
  - `ui.py`: Streamlit-based web UI with tabs for Chat, Files, Models, Logs, Knowledge Map. Features include chat history, selective knowledge bases, retrieval settings, contextual insights, resource monitoring, and export to Markdown.

- **Ingestion Module** (`ingestion/`):
  - `ingestor.py`: Reads files from `raw/` using processors (Text, JSON, PDF, DOCX, HTML). Generates SHA256-based doc IDs, preserves content verbatim, and saves metadata.
  - `pool_manager.py`: Groups documents into "Data Pools" (e.g., for selective querying).
  - `processors.py`: Specialized parsers for different formats using libraries like `pypdf`, `python-docx`, `beautifulsoup4`.

- **Chunking Module** (`chunking/`):
  - `chunker.py`: Splits documents into overlapping chunks (default 500 chars, 50 overlap) with deterministic hashing for stable IDs.

- **Embeddings Module** (`embeddings/`):
  - `embedder.py`: Uses `BAAI/bge-large-en-v1.5` (1024-dim vectors) via SentenceTransformers. Handles GPU/CPU fallback, lazy loading, saves embeddings locally and uploads to Qdrant for vector search.

- **Search Module** (`search/`):
  - `engine.py`: Hybrid search combining Elasticsearch BM25 (lexical) and Qdrant cosine similarity (vector). Loads chunks locally for fallback, indexes in ES and Qdrant during ingestion.
  - `reranker.py`: Uses `BAAI/bge-reranker-v2-m3` cross-encoder for relevance reranking, filtering low-score chunks (threshold -5.0).

- **Graph Module** (`graph/`):
  - `graph_store.py`: NetworkX MultiDiGraph for entities/relations with confidence scores and evidence chunk IDs. Supports expansion via hops for multi-hop reasoning. Integrated with Neo4j for persistent, queryable graph storage.

- **LLM Module** (`llm/`):
  - `client.py`: Interfaces with local `llama.cpp` server (HTTP API at `localhost:8080`). Handles streaming, entity/relation extraction, insight generation, answer grading, and server management (start/stop with subprocess).
  - `manager.py`: Downloads GGUF models from Hugging Face, assigns tasks (e.g., reranking to specific models), and manages metadata.

- **Orchestration Module** (`orchestration/`):
  - `flow.py`: Core logic for document processing, querying, background pooling, and proactive thought evaluation.

- **Utils** (`utils/`):
  - `auditor.py`: System integrity checks (e.g., rebuilds corrupted indexes).
  - `chat_auditor.py`: Logs LLM interactions (requests/responses) to JSONL files.
  - `chat_manager.py`: Persistent chat histories with versioning, insights, and Markdown export.
  - `logger.py`: Centralized logging with timestamps.
  - `resource_monitor.py`: Monitors CPU/RAM/VRAM, estimates model sizes, and enforces resource limits.

## Data Processing Pipeline
1. **Ingestion**: Files from `raw/` are processed into `processed/` with metadata (filename, hash, timestamp, processor type).
2. **Chunking**: Text split into chunks saved as JSON in `processed/chunks/`.
3. **Embedding**: Chunks vectorized and stored as NumPy arrays in `processed/embeddings/`.
4. **Graph Extraction**: LLM extracts entities/relations, stored in `processed/graph.json` (NetworkX format).
5. **Indexing**: Search engine loads chunks/embeddings for hybrid queries.
6. **Query Processing**:
   - Initial hybrid search (lexical + vector) → Rerank → Answer generation with context.
   - Background: Graph expansion + idea pooling (vector search on extracted entities) → Proactive thought evaluation.
7. **Output**: Answers with superscript citations ([Source X]), linked to chunk text/filename/method.

Data structures include:
- `data/pools.json`: JSON with pool names and doc IDs (e.g., `{"0BVIOUS": []}`).
- `processed/graph.json`: Serialized graph (created post-ingestion).
- `raw/`: Input files (e.g., `conversations.json`, `raggedy_overview.txt`, `test.txt` – sample data).
- `logs/`: Audit trails (chat interactions, insights history).
- `models/`: GGUF files (e.g., Gemma, Phi, Qwen) and `metadata.json` for task assignments (e.g., reranking to `bge-reranker-v2-m3-Q4_K_M.gguf`).

## Search and Retrieval Mechanisms
- **Hybrid Search**: Combines BM25 (TF-IDF) for exact matches and vector similarity for semantics. Scores fused for top candidates.
- **Reranking**: Cross-encoder filters irrelevant chunks, ensuring high-quality context.
- **Graph Expansion**: Retrieves chunks linked to query entities via 1-hop neighbors.
- **Idea Pooling**: Embeds extracted entities, searches vectors for additional context.
- **Pooling Trace**: Logs entities, graph expansions, and pooled chunks for transparency.
- **Selective Retrieval**: Users can limit queries to specific pools or docs via UI.

## LLM Integration
- **Server**: Local `llama.cpp` (via `llama-cpp-python[server]==0.3.1`) for inference. Supports streaming, custom prompts, and parameters (temperature, top_p, n_predict).
- **Model Management**: Downloads from Hugging Face, assigns tasks (e.g., embedding to `bge-large-en-v1.5-f16.gguf`). Resource checks prevent OOM.
- **Capabilities**:
  - Answer generation with "omnipotent" prompts (no refusals, embraces hypotheticals).
  - Entity/relation extraction for graph.
  - Insight extraction from conversations.
  - Answer grading (faithfulness, grounding, completeness).
- **Fallbacks**: Mock responses if server offline.

## User Interface
- **CLI** (`main.py`): Simple commands for ingestion, querying, and UI launch.
- **Web UI** (`ui.py`): Streamlit app with:
  - **Chat Tab**: Conversational interface with message versioning, copy/edit, citations (tooltips with source text), continue generation, and background insights (live concepts/terms).
  - **Files Tab**: Data pool management.
  - **Models Tab**: Model download/start/stop server.
  - **Logs Tab**: System logs and insight history, plus **Comprehensive Testing Suite** for verifying all components (ingestion, chunking, embedding, search, reranking, graph, LLM, pooling, resource monitor).
  - **Map Tab**: Knowledge visualization (future).
  - Features: System resource display, selective knowledge bases, retrieval settings (top_k, temperature), export to Markdown, reset functionality.
- **Citations**: Rendered as superscript links with bubble previews of source text.

## Testing and Validation
- **Comprehensive Testing Suite**: Available in the Logs tab of the Web UI, allowing individual and full-system testing of all core components:
  - Ingestion Pipeline (file processing, chunking, embedding, graph extraction)
  - Search Mechanisms (lexical TF-IDF/ES, vector similarity/Qdrant, reranking)
  - LLM Integration (response generation, entity extraction)
  - Graph Operations (knowledge graph loading and querying)
  - Resource Monitoring and Pool Management
  - Service Connectivity (Elasticsearch, Qdrant)
- **Docker Validation**: Future tests will include container startup, service connectivity, and GPU passthrough verification.
- Ensures robustness and verifies functionality after updates or deployments.

## Utilities
- **Auditing**: Integrity checks and repairs.
- **Chat Management**: JSON-based histories with insights, pooled data, traces, and grades.
- **Logging**: File-based logs for debugging.
- **Resource Monitoring**: Prevents GPU overload; prioritizes tasks.
- **Dependencies** (`requirements.txt`): Includes `llama-cpp-python`, `sentence-transformers`, `streamlit`, `networkx`, `scikit-learn`, etc.

## Other Important Aspects
- **Offline-First**: No internet required post-setup; all processing local.
- **Resource Management**: GPU locks, VRAM checks, background task prioritization.
- **Extensibility**: Modular design allows swapping components (e.g., different embedders/rerankers).
- **Challenges Addressed**: Avoids hallucinations via citations; handles ambiguity with probabilistic graphs; supports complex queries via multi-stage retrieval.
- **Setup**: Requires Python 3.8+, CUDA for GPU, `ninja` for builds. Models downloaded via UI.
- **Limitations**: MVP lacks full Docker/Elasticsearch/Qdrant; relies on file storage. Some files (e.g., `processed/graph.json`) created dynamically.
- **Philosophy**: "Soft Probabilistic Graph" allows conflicting interpretations; emphasizes verifiability over dogma.

This system provides a robust foundation for offline knowledge management, balancing performance, accuracy, and usability. For deeper dives, refer to `DOCS/ProjectDesign.md` for the full vision and `PROJECT_STRUCTURE.md` for navigation.