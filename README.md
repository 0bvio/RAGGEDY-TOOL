# RAGGEDY TOOL - MVP Lite

This is a minimally viable product (MVP) of the RAGGEDY TOOL, an offline knowledge intelligence system.

## Setup
1. Ensure Python 3.8+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: For GPU acceleration with `llama-cpp-python`, please follow their [installation guide](https://github.com/abetlen/llama-cpp-python#installation-with-hardware-acceleration-blas-cuda-metal-etc).*

3. Launch the UI to download and manage models:
   ```bash
   python3 main.py ui
   ```

### Docker Setup (Recommended for Full System)
For the complete end-vision system with external services:
1. Ensure Docker and Docker Compose are installed.
2. Build and start the services:
   ```bash
   docker-compose up --build
   ```
3. Access the UI at `http://localhost:8501`.
4. Note: Update the llama-cpp service in `docker-compose.yml` with your desired model path.

## Model Management
RAGGEDY TOOL now supports automated model management:
- **Download**: You can download recommended GGUF models directly from the UI sidebar.
- **Serve**: Start and stop a local LLM server (via `llama-cpp-python`) with one click from the UI.
- **Manual**: You can still use an external `llama.cpp` server at `http://localhost:8080/v1` if preferred.

## Usage
To ingest documents from `data/raw`:
```bash
python3 main.py ingest
```

To ask a question via CLI:
```bash
python3 main.py ask "What is RAGGEDY TOOL?"
```

To launch the UI:
```bash
python3 main.py ui
```

## Project Structure
- `ingestion/`: Handles document intake and metadata tracking.
- `chunking/`: Implements deterministic text splitting.
- `embeddings/`: Generates vector representations of text.
- `search/`: Hybrid search engine (Lexical + Vector).
- `graph/`: Probabilistic knowledge graph storage and expansion.
- `llm/`: Interface for local LLM inference.
- `orchestration/`: Coordinates the reasoning flow and citations.
- `data/`: Persistent storage for raw and processed data.

## Features
- **Offline First**: Runs entirely on your local machine.
- **Hybrid Retrieval**: Combines keyword search with semantic similarity.
- **Concept Pooling**: Multi-stage retrieval that gathers information based on conceptual relationships and conversational history.
- **Knowledge Graph**: Extracts and explores conceptual relationships using probabilistic nodes.
- **Contextual Insights**: Live sidebar tracking concepts, terms, and ideas discussed during your session.
- **Grounded Answers**: Provides superscript citations with instant bubble previews.
- **Knowledge Map**: Interactive visualization of entities and conceptual relationships.
- **Advanced Model Management**: Assign specific GGUF models to background tasks like reranking, insights, or embedding.
- **Inspectable Reasoning**: Trace answers back to their exact source chunks and retrieval methods.
