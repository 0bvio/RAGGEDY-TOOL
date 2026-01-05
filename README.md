# 🤖 RAGGEDY TOOL

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.md)

An offline knowledge intelligence system that ingests unstructured text data, organizes it into concepts and relationships, and provides grounded, citation-backed answers to natural language queries. Built entirely offline using local LLMs and vector databases.

## ✨ Features

- **Offline-First**: Runs entirely on your local machine without internet access
- **Hybrid Retrieval**: Combines lexical search (BM25) with semantic similarity for accurate results
- **Knowledge Graph**: Probabilistic entity-relation extraction with multi-hop reasoning
- **Sentence-Level Citations**: Evidence-backed answers with instant source previews
- **Multi-Stage Retrieval**: Lexical + vector + graph-based context gathering
- **GPU Acceleration**: CUDA support with automatic CPU fallback
- **Model Management**: Automated GGUF model downloads and task assignment
- **Interactive UI**: Streamlit-based interface with chat, file management, and visualization
- **Comprehensive Testing**: Built-in test suite for all components

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker & Docker Compose (for full system)
- NVIDIA GPU (optional, for acceleration)

### Docker Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/raggedy-tool.git
cd raggedy-tool

# Start all services
docker-compose up -d

# Access the UI
open http://localhost:8501
```

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (optional)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

# Launch the UI
python main.py ui
```

## 📖 Usage

### Ingest Documents
Place files in `data/raw/` and run:
```bash
python main.py ingest
```

Supported formats: TXT, PDF, DOCX, HTML, JSON

### Ask Questions
```bash
python main.py ask "What is the main concept discussed in the documents?"
```

### Web Interface
The Streamlit UI provides:
- **Chat**: Conversational interface with citations and insights
- **Files**: Data pool management and selective querying
- **Models**: Download and manage GGUF models
- **Logs**: System monitoring and comprehensive testing
- **Map**: Knowledge graph visualization

## 🏗️ Architecture

### Core Components
- **Ingestion**: Document processing with metadata preservation
- **Chunking**: Deterministic text splitting with overlap
- **Embeddings**: SentenceTransformers for semantic vectors
- **Search**: Hybrid engine (Elasticsearch + Qdrant)
- **Graph**: Neo4j-backed knowledge relationships
- **LLM**: Local inference via llama.cpp
- **Orchestration**: Coordinated reasoning pipeline

### Data Flow
1. Documents → Chunks → Embeddings → Indexed
2. Queries → Hybrid Search → Reranking → Answer Generation
3. Citations linked to source chunks with evidence

### Services (Docker)
- **Elasticsearch**: Lexical search and indexing
- **Qdrant**: Vector database for semantic search
- **Neo4j**: Graph database for knowledge relationships
- **llama.cpp**: Local LLM server
- **Python App**: Orchestration and UI

## 🧪 Testing

The UI includes a comprehensive test suite in the Logs tab to verify:
- Ingestion pipeline (chunking, embedding, graph extraction)
- Search mechanisms (lexical, vector, reranking)
- LLM integration and response generation
- Graph operations and querying
- Service connectivity (Elasticsearch, Qdrant, Neo4j)
- Resource monitoring and pool management

## 📁 Project Structure

```
raggedy-tool/
├── main.py                 # CLI entry point
├── ui.py                   # Streamlit web interface
├── docker-compose.yml      # Multi-service orchestration
├── Dockerfile             # Python app container
├── requirements.txt       # Python dependencies
├── data/
│   ├── raw/              # Input documents
│   ├── processed/        # Chunks, embeddings, graph
│   └── chats/            # Conversation histories
├── models/               # GGUF model files
├── logs/                 # System and audit logs
├── ingestion/            # Document intake modules
├── chunking/             # Text splitting logic
├── embeddings/           # Vector generation
├── search/               # Hybrid search engine
├── graph/                # Knowledge graph storage
├── llm/                  # Local model interface
├── orchestration/        # Pipeline coordination
└── utils/                # Logging, monitoring, auditing
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Run the test suite in the UI Logs tab
5. Commit with clear messages: `git commit -m "Add feature description"`
6. Push and create a pull request

### Development Setup
```bash
# Install in development mode
pip install -e .

# Run tests
# Use the UI test suite or add unit tests
```

## 📄 License

MIT License - see [LICENSE.md](LICENSE.md) for details.

## 🙏 Acknowledgments

Built with:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for local LLM inference
- [SentenceTransformers](https://github.com/UKPLab/sentence-transformers) for embeddings
- [Streamlit](https://streamlit.io/) for the web interface
- [Elasticsearch](https://www.elastic.co/) for lexical search
- [Qdrant](https://qdrant.tech/) for vector search
- [Neo4j](https://neo4j.com/) for graph storage

---

**RAGGEDY TOOL** - Your personal offline research engine and reasoning assistant.
