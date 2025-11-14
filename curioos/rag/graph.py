"""
RAG Pipeline Orchestration Module

This module implements the core RAG (Retrieval-Augmented Generation) workflow using
LangGraph for state management and control flow. It orchestrates the multi-step process
of answering user questions with retrieved context.

Pipeline Stages:
	1. ensure_index: Validate that vector index exists
	2. retrieve: Embed query and search for similar chunks
	3. maybe_refine: Expand search if results have low similarity
	4. generate: Call LLM with context to produce answer

Key features:
- State-based workflow (LangGraph StateGraph)
- Adaptive retrieval (increases top_k if results are poor)
- Automatic line number calculation for citations
- Stateless design (each query is independent)

Design Rationale:
	LangGraph provides explicit control flow for complex multi-step pipelines.
	While overkill for this simple linear flow, it makes it easy to add:
	- Branching logic (e.g., route to different retrievers)
	- Iterative refinement (e.g., multi-hop reasoning)
	- Human-in-the-loop feedback
	- Debugging/observability (state snapshots at each step)

Typical Usage:
	>>> graph = build_graph(vector_store, embedder, groq_client)
	>>> result = graph.invoke({"question": "What is CurioOS?", "top_k": 5})
	>>> print(result["answer"])
"""

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
	"""
	State container for RAG pipeline (currently unused, kept for future extensions).

	LangGraph can use dataclasses for type-safe state, but we use plain dicts
	for simplicity. This dataclass documents the expected state structure.

	Attributes:
		question: User's question string
		contexts: Retrieved context chunks as (file_path, text, start_line, end_line, score) tuples
		answer: Generated answer from LLM
	"""
	question: str
	contexts: List[Tuple[str, str, int, int, float]]
	answer: str


def _line_range_for_text(full_text: str, start: int, end: int) -> Tuple[int, int]:
	"""
	Convert character offsets to line numbers for source citation.

	This helper calculates which lines in the source document a character
	range corresponds to. Used for generating line-level citations.

	Args:
		full_text: Complete text of the document
		start: Character offset where chunk begins
		end: Character offset where chunk ends

	Returns:
		Tuple of (start_line, end_line) as 1-indexed line numbers

	Example:
		>>> text = "Line 1\\nLine 2\\nLine 3"
		>>> _line_range_for_text(text, 7, 13)
		(2, 2)  # "Line 2" is on line 2

	Note:
		Currently not used in the pipeline because we store chunks without
		the full document text. Kept for future enhancement.
	"""
	# Count newlines before the chunk to find start line
	prefix = full_text[:start]
	span = full_text[start:end]
	start_line = prefix.count("\n") + 1  # 1-indexed
	end_line = start_line + span.count("\n")
	return start_line, end_line


def build_graph(store: VectorStore, embedder: Embedder, groq: GroqClient):
	"""
	Build and compile the RAG pipeline as a LangGraph StateGraph.

	This function constructs a directed graph representing the RAG workflow.
	Each node is a function that transforms the state, and edges define
	the execution order.

	Args:
		store: Vector store for similarity search
		embedder: Embedding model for query encoding
		groq: LLM client for answer generation

	Returns:
		Compiled StateGraph ready for invocation

	Graph Structure:
		ensure_index → retrieve → maybe_refine → generate → END

	State Dictionary:
		Input:
			- question: str (user's question)
			- top_k: int (number of chunks to retrieve, default 5)
		Output:
			- contexts: List[Tuple[str, str, int, int, float]] (retrieved chunks)
			- answer: str (LLM-generated response)

	Example:
		>>> graph = build_graph(store, embedder, groq)
		>>> result = graph.invoke({"question": "What is Python?", "top_k": 5})
		>>> print(result["answer"])
		Python is a programming language [1]...

		Sources:
		[1] python_intro.md:1-5
	"""
	# Create StateGraph with dict state (flexible, untyped)
	graph = StateGraph(dict)

	def ensure_index(state: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Validate that vector index exists and is initialized.

		This node runs first to catch errors early if the index is empty
		or corrupted. It calls store.ensure_manifest() which creates the
		manifest file if missing.

		Args:
			state: Current state dictionary

		Returns:
			Unmodified state (no changes)

		Side Effects:
			May create manifest.json if it doesn't exist
		"""
		store.ensure_manifest()
		return state

	def retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Retrieve relevant document chunks using vector similarity search.

		This node:
		1. Extracts the user's question from state
		2. Encodes it into an embedding vector
		3. Searches the vector store for top-k similar chunks
		4. Formats results with line numbers (approximation)
		5. Adds contexts to state

		Args:
			state: State dict with "question" and optional "top_k"

		Returns:
			State dict with added "contexts" field

		Context Format:
			List of (file_path, text, start_line, end_line, similarity_score) tuples
			- file_path: Absolute path to source document
			- text: Chunk text content
			- start_line: Line number where chunk begins (approximate)
			- end_line: Line number where chunk ends (approximate)
			- similarity_score: Cosine similarity [0, 1]

		Note:
			Line numbers are approximate because we only store chunk text,
			not the full document. We assume each chunk starts at line 1.
		"""
		question = state["question"]

		# Encode the question into an embedding vector
		q_emb = embedder.encode_query(question)

		# Search vector store for top-k most similar chunks
		# top_k can be adjusted by maybe_refine node
		results = store.search(q_emb, top_k=state.get("top_k", 5))

		# Format results as (file_path, text, start_line, end_line, score) tuples
		contexts: List[Tuple[str, str, int, int, float]] = []
		for entry, score in results:
			# Note: entry.text is already the chunk text (we don't store full docs)
			# Line numbers are approximate - we count newlines within the chunk
			full_text = entry.text
			start_line = 1  # Chunks start at line 1 (relative to chunk, not document)
			end_line = full_text.count("\n") + 1

			contexts.append((entry.file_path, entry.text, start_line, end_line, score))

		# Add contexts to state
		state["contexts"] = contexts
		return state

	def maybe_refine(state: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Expand retrieval if initial results have low similarity scores.

		This adaptive retrieval node checks if the best result has a low
		similarity score (< 0.35), indicating the query may not have good
		matches. If so, it increases top_k and re-runs retrieval.

		Args:
			state: State dict with "contexts" from retrieve node

		Returns:
			State dict, possibly with updated "contexts" if re-retrieval occurred

		Adaptive Logic:
			- If max(similarity_scores) < 0.35 → increase top_k by 3, re-retrieve
			- Otherwise → pass through unchanged

		Rationale:
			Low similarity means the question may be tangentially related to
			documents. Retrieving more chunks gives the LLM more context to
			work with, potentially finding a weak match.

		Example:
			Initial retrieval: top_k=5, max_score=0.28 (low!)
			→ Increase to top_k=10, re-retrieve
			→ New max_score=0.41 (better)
			→ Continue to generation with 10 chunks
		"""
		# Get current contexts from previous retrieve call
		contexts: List[Tuple[str, str, int, int, float]] = state.get("contexts", [])

		# Check if we have low-quality results
		# If no contexts or all scores < 0.35, expand search
		if not contexts or max(c[-1] for c in contexts) < 0.35:
			# Increase top_k (cap at 10 to avoid overwhelming the LLM)
			state["top_k"] = min(10, state.get("top_k", 5) + 3)
			# Re-run retrieval with expanded top_k
			return retrieve(state)

		# Results are good enough, continue with current contexts
		return state

	def generate(state: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Generate answer using LLM with retrieved context.

		This final node:
		1. Extracts question and contexts from state
		2. Formats them into LLM messages (via prompts.py)
		3. Calls Groq LLM to generate answer
		4. Adds answer to state

		Args:
			state: State dict with "question" and "contexts"

		Returns:
			State dict with added "answer" field

		LLM Interaction:
			Messages = [system prompt, user message with context + question]
			System prompt enforces: citation discipline, conciseness, "I don't know"
			Response should include inline citations [1], [2] and source list at end

		Example Flow:
			Input state: {"question": "What is CurioOS?", "contexts": [...]}
			↓
			Build messages: [
				{"role": "system", "content": "You are CurioOS. Answer concisely..."},
				{"role": "user", "content": "Context:\\n[1] file.txt...\\n\\nQuestion: What is CurioOS?"}
			]
			↓
			Call groq.generate(messages)
			↓
			Output state: {"question": ..., "contexts": ..., "answer": "CurioOS is..."}
		"""
		question = state["question"]
		contexts: List[Tuple[str, str, int, int, float]] = state.get("contexts", [])

		# Build LLM messages with system prompt and formatted context
		messages = build_messages(question, contexts)

		# Generate answer using Groq LLM
		answer = groq.generate(messages)

		# Add answer to state
		state["answer"] = answer
		return state

	# Register all nodes in the graph
	graph.add_node("ensure_index", ensure_index)
	graph.add_node("retrieve", retrieve)
	graph.add_node("maybe_refine", maybe_refine)
	graph.add_node("generate", generate)

	# Define execution flow: ensure_index → retrieve → maybe_refine → generate → END
	graph.set_entry_point("ensure_index")
	graph.add_edge("ensure_index", "retrieve")
	graph.add_edge("retrieve", "maybe_refine")
	graph.add_edge("maybe_refine", "generate")
	graph.add_edge("generate", END)

	# Compile the graph into a runnable
	# This validates the graph structure and prepares it for execution
	return graph.compile()
