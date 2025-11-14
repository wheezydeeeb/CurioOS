from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from langgraph.graph import END, StateGraph  # type: ignore

from ..index.vector_store import VectorStore, IndexEntry
from ..index.embeddings import Embedder
from ..llm.groq_client import GroqClient
from .prompts import build_messages


@dataclass
class RagState:
	question: str
	contexts: List[Tuple[str, str, int, int, float]]
	answer: str


def _line_range_for_text(full_text: str, start: int, end: int) -> Tuple[int, int]:
	prefix = full_text[:start]
	span = full_text[start:end]
	start_line = prefix.count("\n") + 1
	end_line = start_line + span.count("\n")
	return start_line, end_line


def build_graph(store: VectorStore, embedder: Embedder, groq: GroqClient):
	graph = StateGraph(dict)

	def ensure_index(state: Dict[str, Any]) -> Dict[str, Any]:
		store.ensure_manifest()
		return state

	def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
		question = state["question"]
		q_emb = embedder.encode_query(question)
		results = store.search(q_emb, top_k=state.get("top_k", 5))

		contexts: List[Tuple[str, str, int, int, float]] = []
		for entry, score in results:
			# entry.text already holds the chunk
			full_text = entry.text  # best-effort for line numbers (already chunk); we approximate
			start_line, end_line = 1, full_text.count("\n") + 1
			contexts.append((entry.file_path, entry.text, start_line, end_line, score))

		state["contexts"] = contexts
		return state

	def maybe_refine(state: Dict[str, Any]) -> Dict[str, Any]:
		# If we have very low similarity across the board, try to expand retrieval
		contexts: List[Tuple[str, str, int, int, float]] = state.get("contexts", [])
		if not contexts or max(c[-1] for c in contexts) < 0.35:
			state["top_k"] = min(10, state.get("top_k", 5) + 3)
			return retrieve(state)
		return state

	def generate(state: Dict[str, Any]) -> Dict[str, Any]:
		question = state["question"]
		contexts: List[Tuple[str, str, int, int, float]] = state.get("contexts", [])
		messages = build_messages(question, contexts)
		answer = groq.generate(messages)
		state["answer"] = answer
		return state

	graph.add_node("ensure_index", ensure_index)
	graph.add_node("retrieve", retrieve)
	graph.add_node("maybe_refine", maybe_refine)
	graph.add_node("generate", generate)

	graph.set_entry_point("ensure_index")
	graph.add_edge("ensure_index", "retrieve")
	graph.add_edge("retrieve", "maybe_refine")
	graph.add_edge("maybe_refine", "generate")
	graph.add_edge("generate", END)

	return graph.compile()


