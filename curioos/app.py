"""
CurioOS Main Application Entry Point

This is the primary module for running Curio OS. It orchestrates all components and provides
three modes of operation:
1. Index mode (--index): Rebuild the vector index from scratch
2. CLI mode (--ask "question"): Ask a single question and exit
3. Interactive mode (default): Terminal-based Q&A loop with file watching

Architecture Overview:
	┌─────────────┐
	│   User      │
	└──────┬──────┘
	       │
	┌──────▼──────┐     ┌──────────────┐
	│   app.py    ├────►│  config.py   │  Load configuration
	└──────┬──────┘     └──────────────┘
	       │
	       ├───►┌─────────────────┐
	       │    │  index/embedder │  Text → Vectors
	       │    └─────────────────┘
	       │
	       ├───►┌──────────────────┐
	       │    │ index/chroma_store│ Store & Search (ChromaDB)
	       │    └──────────────────┘
	       │
	       ├───►┌─────────────────┐
	       │    │  llm/groq_client│  LLM Generation
	       │    └─────────────────┘
	       │
	       ├───►┌─────────────────┐
	       │    │   rag/graph     │  Pipeline Orchestration
	       │    └─────────────────┘
	       │
	       └───►┌─────────────────┐
	            │ ingest/watcher  │  File Monitoring
	            └─────────────────┘

Key Responsibilities:
- Initialize all components (embedder, vector store, LLM, RAG graph)
- Parse command-line arguments
- Orchestrate indexing workflow
- Handle file change events from watcher
- Provide interactive terminal interface
- Graceful shutdown on Ctrl+C

Typical Usage:
	# Index documents
	$ python -m curioos.app --index

	# Ask single question
	$ python -m curioos.app --ask "What is CurioOS?"

	# Interactive mode
	$ python -m curioos.app
	> Question: What is Python?
	> ...
"""

from __future__ import annotations

import argparse
import logging as py_logging
from pathlib import Path
from typing import List, Tuple

from .config import load_config
from .logging import init_logging
from .ingest.parser import load_and_normalize
from .ingest.chunker import chunk_text
from .ingest.watcher import VaultWatcher
from .index.embeddings import Embedder
from .index.chroma_store import ChromaVectorStore
from .llm.groq_client import GroqClient
from .rag.graph import build_graph
from .rag.prompts import build_messages


# Module-level logger for application events
log = py_logging.getLogger("curioos.app")


def _index_file(path: Path, store: ChromaVectorStore, embedder: Embedder) -> None:
	"""
	Index a single document file into the vector store.

	This function implements the complete indexing pipeline for one file:
	1. Parse and normalize the file text
	2. Split text into overlapping chunks
	3. Generate embeddings for all chunks
	4. Upsert chunks into vector store (replacing old versions)

	Args:
		path: Path to document file (.txt, .md, or .pdf)
		store: Vector store to update
		embedder: Embedding model for generating vectors

	Side Effects:
		- Updates vector store with new/updated chunks
		- Logs indexing progress

	Error Handling:
		- If file has no content → removes any old index entries for this file
		- If file can't be parsed → exception propagates to caller

	Example:
		>>> _index_file(Path("vault/notes.md"), store, embedder)
		INFO:curioos.app:Indexed vault/notes.md with 5 chunks
	"""
	# Step 1: Parse file and normalize text
	text, md5 = load_and_normalize(path)

	# Step 2: Chunk the text into overlapping segments
	chunks = chunk_text(text, chunk_size=800, overlap=150)

	# Handle empty files (no content after normalization)
	if not chunks:
		# Remove any old content for this file from the index
		store.remove_file(path)
		log.info("No content in %s; removed from index if any", path)
		return

	# Step 3: Generate embeddings for all chunk texts
	# Extract just the text from each (start, end, text) tuple
	embeddings = embedder.encode_texts([ch[2] for ch in chunks])

	# Step 4: Upsert chunks into vector store
	# This replaces any existing chunks for this file
	store.upsert_chunks(path, md5, chunks, embeddings)

	log.info("Indexed %s with %d chunks", path, len(chunks))


def _reindex_all(vault_dir: Path, store: ChromaVectorStore, embedder: Embedder) -> None:
	"""
	Rebuild the entire vector index from scratch.

	This function scans the vault directory recursively and indexes all
	supported document files (.txt, .md, .pdf). Used for initial setup
	and full index rebuilds (--index flag).

	Args:
		vault_dir: Root directory containing documents
		store: Vector store to populate
		embedder: Embedding model for generating vectors

	Side Effects:
		- Replaces all chunks in vector store
		- Logs progress for each file
		- May take several minutes for large vaults

	Example:
		>>> _reindex_all(Path("data/vault"), store, embedder)
		INFO:curioos.app:Indexed vault/doc1.txt with 3 chunks
		INFO:curioos.app:Indexed vault/doc2.md with 5 chunks
		INFO:curioos.app:Reindexed 2 files
	"""
	count = 0
	# Recursively find all supported document files
	for p in vault_dir.rglob("*"):
		if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}:
			_index_file(p, store, embedder)
			count += 1

	log.info("Reindexed %d files", count)


def _on_change(path: Path, kind: str, store: ChromaVectorStore, embedder: Embedder) -> None:
	"""
	Handle file system change events from VaultWatcher.

	This callback is invoked by the file watcher when documents are
	created, modified, or deleted. It updates the vector index accordingly.

	Args:
		path: Path to changed file
		kind: Event type - "created", "modified", or "deleted"
		store: Vector store to update
		embedder: Embedding model for generating vectors

	Event Handling:
		- "deleted" → remove all chunks for this file
		- "created" or "modified" → re-index the file

	Error Handling:
		Catches all exceptions to prevent watcher thread from crashing.
		Logs warnings for failed operations but continues monitoring.

	Example:
		>>> _on_change(Path("vault/doc.txt"), "modified", store, embedder)
		INFO:curioos.app:Indexed vault/doc.txt with 4 chunks
	"""
	try:
		if kind == "deleted":
			# File was deleted - remove from index
			store.remove_file(path)
			log.info("Removed %s from index", path)
		else:
			# File was created or modified - re-index it
			_index_file(path, store, embedder)
	except Exception as e:
		# Don't crash the watcher thread on errors
		# Log the error and continue monitoring
		log.warning("Failed to process change for %s: %s", path, e)


def main() -> None:
	"""
	Main entry point for CurioOS application.

	This function:
	1. Loads configuration from environment
	2. Initializes logging
	3. Creates embedder and vector store
	4. Parses command-line arguments
	5. Routes to appropriate mode (index, CLI, or interactive)

	Command-Line Interface:
		python -m curioos.app [--index] [--ask "question"]

		--index: Rebuild vector index from scratch
		--ask "question": Ask single question and exit
		(no args): Interactive terminal mode with file watching

	Exit Codes:
		0: Normal exit
		(Exception): Abnormal termination with stack trace
	"""
	# Step 1: Load configuration from .env and environment variables
	cfg = load_config()

	# Step 2: Initialize logging with configured level
	init_logging(cfg.log_level)
	log.info("CurioOS starting")

	# Step 3: Initialize core components

	# Embedder: Converts text → vectors
	# Downloads model on first run if not cached
	embedder = Embedder(cfg.embed_model, cache_folder=cfg.models_cache_dir)

	# Vector Store: Stores embeddings and provides similarity search
	# Loads existing index from disk if available
	store = ChromaVectorStore(cfg.index_dir, embedder.model_name)
	store.ensure_manifest()

	# Step 4: Parse command-line arguments
	parser = argparse.ArgumentParser(description="CurioOS")
	parser.add_argument("--index", action="store_true", help="Rebuild index for the vault directory")
	parser.add_argument("--ask", type=str, help="Ask a question via CLI and print the answer")
	args = parser.parse_args()

	# Step 5: Handle --index mode (rebuild index and exit)
	if args.index:
		_reindex_all(cfg.vault_dir, store, embedder)
		log.info("Index rebuild complete")
		return

	# Step 6: Initialize LLM and RAG pipeline for Q&A modes
	# (Both --ask and interactive mode need these)

	# Groq LLM client
	groq = GroqClient(cfg.groq_api_key, cfg.groq_model)

	# RAG pipeline (LangGraph)
	graph = build_graph(store, embedder, groq)

	def ask(question: str) -> str:
		"""
		Ask a question and get an answer using the RAG pipeline.

		This helper function wraps graph invocation for convenience.
		It initializes the state with the question and top_k, runs
		the pipeline, and extracts the answer.

		Args:
			question: User's question string

		Returns:
			LLM-generated answer with citations

		Pipeline Flow:
			question → embed → retrieve → refine → generate → answer
		"""
		# Initialize state with question and default top_k
		state = {"question": question, "top_k": 5}

		# Run the RAG graph (ensure_index → retrieve → maybe_refine → generate)
		result = graph.invoke(state)

		# Extract answer from final state
		return result.get("answer", "(no answer)")

	# Step 7: Handle --ask mode (single question via CLI)
	if args.ask:
		print(ask(args.ask))
		return

	# Step 8: Interactive mode (terminal Q&A loop with file watching)

	# Start file watcher for automatic re-indexing
	# Lambda creates a closure that captures store and embedder
	watcher = VaultWatcher(cfg.vault_dir, lambda p, k: _on_change(p, k, store, embedder))
	watcher.start()
	log.info("Watcher started for %s", cfg.vault_dir)

	# Display welcome banner
	log.info("CurioOS ready. Type your questions (or 'exit' to quit)")
	print(f"\n{'='*60}")
	print(f"CurioOS Terminal - Model: {cfg.groq_model}")
	print(f"Vault: {cfg.vault_dir}")
	print(f"{'='*60}\n")

	# Main interaction loop
	try:
		while True:
			try:
				# Prompt for question
				question = input("\n> Question: ").strip()

				# Skip empty input
				if not question:
					continue

				# Handle exit commands
				if question.lower() in ('exit', 'quit', 'q'):
					print("Goodbye!")
					break

				# Show "thinking" indicator while processing
				print("\nThinking...", end='', flush=True)

				# Run RAG pipeline
				answer = ask(question)

				# Clear "Thinking..." message
				print("\r" + " " * 15 + "\r", end='')

				# Display answer
				print(f"\nAnswer:\n{answer}\n")

			except KeyboardInterrupt:
				# Handle Ctrl+C gracefully during input
				print("\n\nUse 'exit' to quit.")
				continue

	finally:
		# Cleanup: stop file watcher
		watcher.stop()
		log.info("CurioOS stopped")


if __name__ == "__main__":
	# Entry point when running as a module: python -m curioos.app
	main()