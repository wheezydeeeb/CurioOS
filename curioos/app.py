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
from .index.vector_store import VectorStore
from .llm.groq_client import GroqClient
from .rag.graph import build_graph
from .rag.prompts import build_messages


log = py_logging.getLogger("curioos.app")


def _index_file(path: Path, store: VectorStore, embedder: Embedder) -> None:
	text, md5 = load_and_normalize(path)
	chunks = chunk_text(text, chunk_size=800, overlap=150)
	if not chunks:
		# Remove any old content for this file
		store.remove_file(path)
		log.info("No content in %s; removed from index if any", path)
		return
	embeddings = embedder.encode_texts([ch[2] for ch in chunks])
	store.upsert_chunks(path, md5, chunks, embeddings)
	log.info("Indexed %s with %d chunks", path, len(chunks))


def _reindex_all(vault_dir: Path, store: VectorStore, embedder: Embedder) -> None:
	count = 0
	for p in vault_dir.rglob("*"):
		if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}:
			_index_file(p, store, embedder)
			count += 1
	log.info("Reindexed %d files", count)


def _on_change(path: Path, kind: str, store: VectorStore, embedder: Embedder) -> None:
	try:
		if kind == "deleted":
			store.remove_file(path)
			log.info("Removed %s from index", path)
		else:
			_index_file(path, store, embedder)
	except Exception as e:
		log.warning("Failed to process change for %s: %s", path, e)


def main() -> None:
	cfg = load_config()
	init_logging(cfg.log_level)
	log.info("CurioOS starting")

	embedder = Embedder(cfg.embed_model, cache_folder=cfg.models_cache_dir)
	store = VectorStore(cfg.index_dir, embedder.model_name)
	store.ensure_manifest()

	parser = argparse.ArgumentParser(description="CurioOS")
	parser.add_argument("--index", action="store_true", help="Rebuild index for the vault directory")
	parser.add_argument("--ask", type=str, help="Ask a question via CLI and print the answer")
	args = parser.parse_args()

	if args.index:
		_reindex_all(cfg.vault_dir, store, embedder)
		log.info("Index rebuild complete")
		return

	# Prepare LLM and graph
	groq = GroqClient(cfg.groq_api_key, cfg.groq_model)
	graph = build_graph(store, embedder, groq)

	def ask(question: str) -> str:
		state = {"question": question, "top_k": 5}
		result = graph.invoke(state)
		return result.get("answer", "(no answer)")

	if args.ask:
		print(ask(args.ask))
		return

	# Start watcher
	watcher = VaultWatcher(cfg.vault_dir, lambda p, k: _on_change(p, k, store, embedder))
	watcher.start()
	log.info("Watcher started for %s", cfg.vault_dir)

	# Terminal-only interactive mode
	log.info("CurioOS ready. Type your questions (or 'exit' to quit)")
	print(f"\n{'='*60}")
	print(f"CurioOS Terminal - Model: {cfg.groq_model}")
	print(f"Vault: {cfg.vault_dir}")
	print(f"{'='*60}\n")

	try:
		while True:
			try:
				question = input("\n> Question: ").strip()
				if not question:
					continue
				if question.lower() in ('exit', 'quit', 'q'):
					print("Goodbye!")
					break

				print("\nThinking...", end='', flush=True)
				answer = ask(question)
				print("\r" + " " * 15 + "\r", end='')  # Clear "Thinking..."
				print(f"\nAnswer:\n{answer}\n")
			except KeyboardInterrupt:
				print("\n\nUse 'exit' to quit.")
				continue
	finally:
		watcher.stop()
		log.info("CurioOS stopped")


if __name__ == "__main__":
	main()


