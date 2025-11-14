# CurioOS Architecture Documentation

This document provides detailed technical documentation for developers who want to understand, modify, or extend CurioOS.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Module Architecture](#module-architecture)
3. [Data Flow](#data-flow)
4. [Component Details](#component-details)
5. [Design Decisions](#design-decisions)
6. [Extension Points](#extension-points)
7. [Performance Considerations](#performance-considerations)

---

## System Overview

CurioOS is a **Retrieval-Augmented Generation (RAG) system** that enables natural language question-answering over a local document collection. It combines:

- **Local vector search** (NumPy + scikit-learn)
- **Semantic embeddings** (Sentence Transformers)
- **LLM generation** (Groq API)
- **Workflow orchestration** (LangGraph)

### Core Architecture Principles

1. **Local-First**: All document data and embeddings stored locally
2. **Modular Design**: Clear separation of concerns between components
3. **Simplicity Over Scale**: Optimized for <10k chunks, not millions
4. **Type Safety**: Heavy use of type hints and dataclasses
5. **Extensibility**: Easy to swap components (embedders, LLMs, stores)

---

## Module Architecture

```
curioos/
   app.py                  # Application orchestration & CLI
   config.py               # Configuration management
   logging.py              # Logging setup
   ingest/                 # Document ingestion pipeline
      parser.py           # File parsing & normalization
      chunker.py          # Text chunking with overlap
      watcher.py          # File system monitoring
   index/                  # Vector indexing & search
      embeddings.py       # Sentence transformer wrapper
      vector_store.py     # Local vector database
   llm/                    # LLM integration
      groq_client.py      # Groq API wrapper
   rag/                    # RAG pipeline
       graph.py            # LangGraph workflow
       prompts.py          # Prompt engineering
```

### Module Responsibilities

| Module | Responsibility | Key Classes/Functions |
|--------|---------------|----------------------|
| `config.py` | Load environment variables, provide defaults | `AppConfig`, `load_config()` |
| `logging.py` | Configure rotating file logs | `init_logging()` |
| `ingest/parser.py` | Parse .txt/.md/.pdf files | `load_and_normalize()` |
| `ingest/chunker.py` | Split text into overlapping chunks | `chunk_text()` |
| `ingest/watcher.py` | Monitor vault for changes | `VaultWatcher` |
| `index/embeddings.py` | Generate embeddings | `Embedder` |
| `index/vector_store.py` | Store & search embeddings | `VectorStore` |
| `llm/groq_client.py` | Call Groq LLM | `GroqClient` |
| `rag/graph.py` | Orchestrate RAG workflow | `build_graph()` |
| `rag/prompts.py` | Format prompts for LLM | `build_messages()` |
| `app.py` | CLI, modes, initialization | `main()` |

---

## Data Flow

### Indexing Flow

```
User runs: python -m curioos.app --index
                    |
                    v
 app.py: _reindex_all()
 - Scan vault_dir for .txt/.md/.pdf files
 - Call _index_file() for each
                    |
                    v
 app.py: _index_file()
 1. parser.load_and_normalize(path)
    => (text, md5_hash)
 2. chunker.chunk_text(text)
    => [(start, end, chunk_text), ...]
 3. embedder.encode_texts([chunk_text, ...])
    => np.ndarray (N x 384)
 4. store.upsert_chunks(path, md5, chunks, embs)
                    |
                    v
 vector_store.py: VectorStore.upsert_chunks()
 - Remove old chunks for this file
 - Add new chunks with auto-incremented IDs
 - Save embeddings.npy, index.json, manifest.json
 - Rebuild NearestNeighbors index
```

### Query Flow

```
User asks: "What is Python?"
           |
           v
 app.py: ask()
 - Create state: {"question": "...", "top_k": 5}
 - Invoke graph
           |
           v
 graph.py: LangGraph Pipeline

 Node 1: ensure_index
   - Verify manifest.json exists

 Node 2: retrieve
   - embedder.encode_query(question)
   - store.search(query_emb, top_k=5)
   - Returns: [(entry, score), ...]

 Node 3: maybe_refine
   - If max(scores) < 0.35:
       top_k += 3, re-run retrieve

 Node 4: generate
   - prompts.build_messages(question, contexts)
   - groq.generate(messages)
   - Returns: answer with citations
           |
           v
 app.py: Display answer to user
```

---

## Component Details

### 1. Configuration Management (`config.py`)

**Purpose**: Centralize all configuration in one place

**Key Components**:
- `AppConfig` dataclass: Type-safe configuration container
- `load_config()`: Load from .env + environment + defaults

**Configuration Sources** (priority order):
1. Environment variables
2. .env file (via python-dotenv)
3. Hardcoded defaults

---

### 2. Document Ingestion (`ingest/`)

#### `parser.py` - File Parsing

**Supported Formats**: .txt, .md, .pdf

**Normalization**:
- Standardize newlines (CRLF/CR => LF)
- Collapse excessive blank lines (max 1 consecutive)

#### `chunker.py` - Text Chunking

**Parameters**:
- `chunk_size=800` chars
- `overlap=150` chars

**Strategy**: Paragraph-aware with overlap

#### `watcher.py` - File System Monitoring

**Technology**: watchdog library with debouncing (1-second window)

---

### 3. Vector Indexing (`index/`)

#### `embeddings.py` - Embedding Generation

**Model**: Sentence Transformers (default: all-MiniLM-L6-v2)
**Dimensions**: 384
**Features**: L2-normalized, local caching

#### `vector_store.py` - Local Vector Database

**Storage**:
- `embeddings.npy` - NumPy array (N x 384, float32)
- `index.json` - Metadata (file paths, offsets, text)
- `manifest.json` - Index info (model, count, dimensions)

**Search**: sklearn NearestNeighbors with cosine similarity

---

### 4. LLM Integration (`llm/groq_client.py`)

**Wrapper**: LangChain's ChatGroq
**Parameters**: temperature=0.2, max_tokens=600
**Error Handling**: Graceful fallback if API key missing

---

### 5. RAG Pipeline (`rag/`)

#### `prompts.py` - Prompt Engineering

**System Prompt**: Enforces citation discipline, conciseness, "I don't know" responses

#### `graph.py` - LangGraph Workflow

**Flow**: ensure_index => retrieve => maybe_refine => generate => END

**Adaptive Retrieval**: If max similarity < 0.35, increase top_k and re-retrieve

---

## Design Decisions

### Why Local Vector Store?

**Rationale**: Simplicity, privacy, no external dependencies
**Trade-off**: Limited to ~10k chunks (scalability)

### Why Groq?

**Rationale**: 10-100x faster than OpenAI, competitive pricing
**Trade-off**: Not quite GPT-4 quality

### Why Overlapping Chunks?

**Rationale**: Preserves context at boundaries, improves retrieval
**Trade-off**: ~20% larger index

---

## Extension Points

### Adding New Document Formats

**Location**: `curioos/ingest/parser.py`

Add parsing function and update `load_and_normalize()`

### Swapping Embedding Models

**Location**: `curioos/index/embeddings.py`

Modify `Embedder` class to support OpenAI, Cohere, etc.

### Adding Approximate NN

**Location**: `curioos/index/vector_store.py`

Replace sklearn with FAISS/Annoy for >10k chunks

---

## Performance Considerations

### Indexing Performance

**Bottlenecks**: PDF parsing (~1 sec/page), embedding generation (~0.1 sec/chunk)

### Query Performance

**Typical Latency**: ~515ms total (10ms embed + 5ms search + 500ms LLM)

### Memory Usage

**10k chunks**: ~150 MB RAM (90 MB model + 15 MB index + 30 MB sklearn)

---

**For questions or contributions, see the main [README.md](./README.md).**
