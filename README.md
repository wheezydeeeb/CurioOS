# CurioOS

**CurioOS** is a lightweight, local-first Retrieval-Augmented Generation (RAG) system that lets you ask questions about your personal document collection. It indexes your .txt, .md, and .pdf files locally, enabling fast semantic search powered by vector embeddings and LLMs.

---

## Features

- **Local-First**: All data stays on your machine. No cloud dependencies for storage.
- **Multiple Formats**: Supports .txt, .md, and .pdf documents
- **Real-Time Updates**: Automatically re-indexes when files change
- **Citation Support**: Answers include source references (file:line-range)
- **Fast & Private**: Uses local embeddings + Groq's ultra-fast LLM API
- **Three Modes**: Index-only, single-question CLI, or interactive terminal

---

## Quick Start

### Prerequisites

- Python 3.10+
- Groq API Key (free at [console.groq.com](https://console.groq.com))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/CurioOS.git
   cd CurioOS
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

5. **Add documents to vault**
   ```bash
   # Put your documents in ./data/vault/
   cp ~/Documents/*.txt ./data/vault/
   ```

6. **Build the index**
   ```bash
   python -m curioos.app --index
   ```

7. **Start asking questions!**
   ```bash
   python -m curioos.app
   ```

---

## Usage

### Interactive Mode (default)

Start an interactive terminal session where you can ask multiple questions:

```bash
python -m curioos.app
```

```
============================================================
CurioOS Terminal - Model: llama-3.3-70b-versatile
Vault: /home/user/CurioOS/data/vault
============================================================

> Question: What is Python?

Answer:
Python is a high-level programming language known for its simplicity
and readability [1]. It was created by Guido van Rossum in 1991 [2].

Sources:
[1] python_intro.txt:5-10
[2] history.md:15-18

> Question: exit
Goodbye!
```

### Single Question Mode

Ask one question and exit (useful for scripting):

```bash
python -m curioos.app --ask "What is machine learning?"
```

### Index Rebuild

Rebuild the entire vector index (run after adding many new documents):

```bash
python -m curioos.app --index
```

---

## Configuration

Edit `.env` to customize CurioOS:

```env
# Required: Groq API key
GROQ_API_KEY=gsk_your_key_here

# Optional: Paths
CURIO_VAULT=./data/vault           # Where your documents live
CURIO_INDEX=./data/index           # Where the vector index is stored
CURIO_MODELS_CACHE=./models        # Where embedding models are cached

# Optional: Models
GROQ_MODEL=llama-3.3-70b-versatile # LLM model name
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2  # Embedding model

# Optional: Logging
LOG_LEVEL=INFO                      # DEBUG, INFO, WARNING, ERROR
```

### Supported LLM Models

CurioOS uses Groq for ultra-fast inference. Recommended models:

- **llama-3.3-70b-versatile** (default) - Best balance of speed & quality
- **llama-3.1-8b-instant** - Fastest, good for simple questions
- **mixtral-8x7b-32768** - Large context window (32k tokens)

See [Groq's model list](https://console.groq.com/docs/models) for all options.

### Supported Embedding Models

CurioOS uses Sentence Transformers for local embeddings:

- **all-MiniLM-L6-v2** (default) - 384 dims, fast, good quality
- **all-mpnet-base-v2** - 768 dims, slower but better quality
- **paraphrase-multilingual-MiniLM-L12-v2** - Multilingual support

---

## Architecture Overview

```
┌──────────────┐
│  Documents   │  (.txt, .md, .pdf)
└──────┬───────┘
       │
┌──────▼────────┐
│ Parser &      │  Normalize text, extract content
│ Chunker       │  Split into 800-char overlapping chunks
└──────┬────────┘
       │
┌──────▼────────┐
│  Embedder     │  Generate 384-dim vectors
└──────┬────────┘  (sentence-transformers/all-MiniLM-L6-v2)
       │
┌──────▼────────┐
│ Vector Store  │  Local numpy arrays + JSON
│ (embeddings.  │  K-nearest neighbors search
│  npy, index.  │  (cosine similarity)
│  json)        │
└──────┬────────┘
       │
       │  User asks question
       │
┌──────▼────────┐
│  RAG Pipeline │  1. Embed query
│ (LangGraph)   │  2. Search top-k chunks
│               │  3. Format context
│               │  4. Call LLM (Groq)
└──────┬────────┘
       │
┌──────▼────────┐
│   Answer      │  with inline citations [1], [2]
└───────────────┘
```

For detailed architecture documentation, see [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## Project Structure

```
CurioOS/
├── curioos/                    # Main package
│   ├── app.py                  # Entry point & CLI
│   ├── config.py               # Configuration management
│   ├── logging.py              # Logging setup
│   ├── ingest/                 # Document ingestion
│   │   ├── parser.py           # File parsing (.txt, .md, .pdf)
│   │   ├── chunker.py          # Text chunking with overlap
│   │   └── watcher.py          # File system monitoring
│   ├── index/                  # Vector indexing
│   │   ├── embeddings.py       # Sentence transformer wrapper
│   │   └── vector_store.py     # Local vector database
│   ├── llm/                    # LLM integration
│   │   └── groq_client.py      # Groq API wrapper
│   └── rag/                    # RAG pipeline
│       ├── graph.py            # LangGraph workflow
│       └── prompts.py          # Prompt templates
├── data/                       # Data directories
│   ├── vault/                  # Your documents (gitignored)
│   └── index/                  # Vector index (gitignored)
├── models/                     # Cached embedding models (gitignored)
├── .env                        # Configuration (gitignored)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## How It Works

### 1. Indexing

When you run `--index` or modify a document:

1. **Parse**: Extract text from .txt/.md/.pdf files
2. **Normalize**: Standardize newlines, collapse whitespace
3. **Chunk**: Split into 800-char overlapping segments (preserves context)
4. **Embed**: Generate 384-dim vectors using sentence-transformers
5. **Store**: Save embeddings + metadata to `data/index/`

### 2. Querying

When you ask a question:

1. **Embed**: Convert question to 384-dim vector
2. **Search**: Find top-5 most similar chunks (cosine similarity)
3. **Refine**: If similarity < 0.35, expand to top-10 (adaptive retrieval)
4. **Generate**: Send context + question to Groq LLM
5. **Cite**: LLM includes source references in answer

### 3. File Watching

In interactive mode, CurioOS monitors your vault:

- **Created/Modified** → Re-index the file automatically
- **Deleted** → Remove from index
- **Debouncing** → Groups rapid changes (autosave, etc.) into one update

---

## Troubleshooting

### "Groq API key not configured"

Add your API key to `.env`:
```env
GROQ_API_KEY=gsk_your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### "No results found" for questions

1. Check that files are in `./data/vault/`
2. Rebuild the index: `python -m curioos.app --index`
3. Verify index exists: `ls ./data/index/` (should have `embeddings.npy`, `index.json`, `manifest.json`)

### PDF files not indexing

- Only text-based PDFs are supported (not scanned images)
- Try converting scanned PDFs with OCR first

### Memory errors with large vaults

- Current implementation loads entire index into RAM
- For >10k chunks, consider:
  - Using smaller embedding model
  - Splitting vault into multiple instances
  - Using approximate nearest neighbors (FAISS, Annoy)

---

## Limitations

- **PDF Support**: Text-only (no OCR for scanned images)
- **Scalability**: Entire index must fit in RAM (~10k chunks max)
- **Single-User**: No multi-user support or access control
- **No Streaming**: Answers are generated in one shot (not streamed)
- **English-Optimized**: Default embedding model works best for English

---

## Future Enhancements

- [ ] Add OCR support for scanned PDFs
- [ ] Implement streaming LLM responses
- [ ] Add web UI (Gradio/Streamlit)
- [ ] Support for images and tables
- [ ] Multi-language embedding models
- [ ] Query history and analytics
- [ ] Export conversations to markdown

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **LangGraph** - RAG workflow orchestration
- **Groq** - Ultra-fast LLM inference
- **Sentence Transformers** - Local text embeddings
- **scikit-learn** - K-nearest neighbors search
- **pdfminer.six** - PDF text extraction
- **watchdog** - File system monitoring

---

**Happy exploring with CurioOS!** 🔍🤖
