# CurioOS v0.2

CurioOS is a lightweight Python app for instant Q&A over a local folder of documents. Drop `.txt`, `.md`, or `.pdf` files into your vault, and ask questions in the terminal. Answers are generated with Groq and a local retrieval index powered by SentenceTransformers.

## Features
- Local indexing of `.txt`, `.md`, `.pdf`
- Retrieval-augmented generation (RAG) using Groq via LangGraph
- Terminal-based interactive interface
- Background folder watcher to auto-update the index
- Local model caching (models stored in `./models` directory)

## Quick Start
1. Install Python 3.10+
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or .venv\Scripts\activate on Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set your `GROQ_API_KEY` in `.env` file
5. Run the app:
   ```bash
   python -m curioos.app
   ```
6. Type your questions and press Enter. Type `exit` to quit.

## Configuration
Environment variables (via `.env`):
- `GROQ_API_KEY`: Your Groq API key
- `CURIO_VAULT`: Path to your documents folder (default: `./data/vault`)
- `CURIO_INDEX`: Path to index folder (default: `./data/index`)
- `CURIO_MODELS_CACHE`: Path to cache embedding models (default: `./models`)
- `EMBED_MODEL`: SentenceTransformers model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `GROQ_MODEL`: Groq LLM (default: `llama-3.3-70b-versatile`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

## CLI Smoke Test
You can run ingestion and ask a question via CLI:
```bash
python -m curioos.app --index
python -m curioos.app --ask "What is this project about?"
```

## Notes
- No external vector DB is required. All embeddings and metadata are stored locally under `data/index`.
- PDF parsing uses `pdfminer.six`. Scanned PDFs (images) are not supported.
- Embedding models are downloaded once and cached locally in the `./models` directory.
- The app watches your vault folder in the background and automatically re-indexes files when they change.

## License
MIT


