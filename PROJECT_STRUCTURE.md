# RAGGEDY TOOL - Project Structure

This document provides a comprehensive overview of the project's file structure to assist AI systems and developers in navigating the codebase.

## Root Directory
- `main.py`: The entry point for the application. Supports commands for ingestion, querying via CLI, and launching the UI.
- `ui.py`: The Streamlit-based user interface.
- `requirements.txt`: Python dependencies.
- `userinstruction.md`: Detailed guide for system-level dependencies (CUDA, Ninja, etc.).
- `README.md`: General project overview and setup instructions.
- `PROJECT_STRUCTURE.md`: This file.
- `.gitignore`: Rules for excluding local/sensitive data from version control.

## Directories

### `ingestion/`
Handles the intake of raw documents.
- `ingestor.py`: Core logic for reading files from `data/raw` and saving to `data/processed`.
- `processors.py`: Specialized classes for parsing different file formats (PDF, DOCX, JSON, HTML, Text).
- `pool_manager.py`: Manages groupings of documents into "Data Pools".

### `chunking/`
Responsible for breaking down large documents into manageable pieces.
- `chunker.py`: Implements deterministic text splitting with context overlap.

### `embeddings/`
Generates vector representations of text.
- `embedder.py`: Uses `BAAI/bge-large-en-v1.5` to create high-dimensional embeddings.

### `search/`
The retrieval engine.
- `engine.py`: Implements hybrid search (Lexical BM25-style + Vector similarity).
- `reranker.py`: Uses `BAAI/bge-reranker-v2-m3` to refine search results for relevance.

### `graph/`
Manages the conceptual knowledge layer.
- `graph_store.py`: Implementation of a probabilistic knowledge graph using NetworkX.

### `llm/`
Interfaces for Large Language Model interactions.
- `client.py`: Handles communication with the local LLM server (compatible with `llama.cpp`).
- `manager.py`: Manages local GGUF model files, nicknames, and downloads from Hugging Face.

### `orchestration/`
Coordinates the multi-layered RAG pipeline.
- `flow.py`: The `RaggedyOrchestrator` class which ties ingestion, search, graph, and LLM layers together.

### `utils/`
Common utility modules.
- `logger.py`: Centralized logging system for all components.
- `chat_manager.py`: Manages persistent, multi-session chat histories.
- `resource_monitor.py`: Tracks CPU, RAM, and VRAM usage for safe, prioritized model operations.
- `auditor.py`: Performs system-wide integrity checks (Audit & Self-Repair).

### `data/` (Excluded from Git)
Persistent storage for the system.
- `raw/`: Directory for input documents.
- `processed/`: Contains extracted text, chunks, and embeddings.
- `chats/`: JSON files for saved conversation history.
- `pools.json`: Metadata for data groupings.

### `models/` (Excluded from Git)
Storage for local LLM files.
- `*.gguf`: Large language model weights.
- `metadata.json`: Model nicknames and download info.

### `logs/` (Excluded from Git)
Diagnostic information.
- `*.log`: System logs and LLM server output records.
