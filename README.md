# CurioOS v0.1

CurioOS is a lightweight Windows Python app for instant Q&A over a local folder of documents. Drop `.txt`, `.md`, or `.pdf` files into your vault, press a global hotkey, and ask questions grounded in your files. Answers are generated with Groq and a local retrieval index powered by SentenceTransformers.

## Features
- Local indexing of `.txt`, `.md`, `.pdf`
- Retrieval-augmented generation (RAG) using Groq via LangGraph
- Small always-on-top popup UI triggered by a global hotkey
- Background folder watcher to auto-update the index

## Quick Start
1. Install Python 3.10+ on Windows.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and set `GROQ_API_KEY`.
5. Run the app:
   ```bash
   python -m curioos.app
   ```
6. Press Ctrl+Shift+Space to open/close the popup. Type a question and press Enter.

## Configuration
Environment variables (via `.env`):
- `GROQ_API_KEY`: Your Groq API key
- `CURIO_VAULT`: Path to your documents folder (default: `./data/vault`)
- `CURIO_INDEX`: Path to index folder (default: `./data/index`)
- `HOTKEY`: Global hotkey (default: `ctrl+shift+space`)
- `EMBED_MODEL`: SentenceTransformers model (default: `sentence-transformers/all-MiniLM-L6-v2`) 
- `GROQ_MODEL`: Groq LLM (default: `llama-3.3-70b-versatile`)

## CLI Smoke Test
You can run ingestion and ask a question via CLI:
```bash
python -m curioos.app --index
python -m curioos.app --ask "What is this project about?"
```

## Notes
- No external vector DB is required. All embeddings and metadata are stored locally under `data/index`.
- PDF parsing uses `pdfminer.six`. Scanned PDFs (images) are not supported in v0.1.

## License
MIT


