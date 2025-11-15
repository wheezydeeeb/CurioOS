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
	>>> graph = build_graph(vector_store, embedder, groq_client, initial_top_k=5)
	>>> result = graph.invoke({"question": "What is CurioOS?"})
	>>> print(result["answer"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langgraph.graph import END, StateGraph  # type: ignore
from langchain_core.messages import BaseMessage  # type: ignore
from pydantic import BaseModel, Field

from ..index.chroma_store import ChromaVectorStore, IndexEntry
from ..index.embeddings import Embedder
from ..llm.groq_client import GroqClient
from .prompts import build_messages


class RagState(BaseModel):
	"""
	Type-safe state container for RAG pipeline.

	LangGraph uses this Pydantic model for type-safe state management throughout
	the pipeline. Fields are progressively populated as the graph executes.

	Attributes:
		question: User's question string (required input)
		contexts: Retrieved context chunks as (file_path, text, start_line, end_line, score) tuples
		          Populated by retrieve node
		answer: Generated answer from LLM (final output)
		        Populated by generate node
	"""
	question: str
	contexts: Optional[List[Tuple[str, str, int, int, float]]] = Field(default=None)
	answer: Optional[str] = Field(default=None)


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


def build_graph(
	store: ChromaVectorStore,
	embedder: Embedder,
	groq: GroqClient,
	initial_top_k: int = 5,
	low_score_threshold: float = 0.35,
	refinement_increase: int = 3,
	max_top_k: int = 10,
):
	"""
	Build and compile the RAG pipeline as a LangGraph StateGraph.

	This function constructs a directed graph representing the RAG workflow.
	Each node is a function that transforms the state, and edges define
	the execution order.

	Args:
		store: Vector store for similarity search
		embedder: Embedding model for query encoding
		groq: LLM client for answer generation
		initial_top_k: Initial number of chunks to retrieve (default: 5)
		low_score_threshold: Similarity threshold below which to expand search (default: 0.35)
		refinement_increase: Number of additional chunks to retrieve on refinement (default: 3)
		max_top_k: Maximum number of chunks to retrieve after refinement (default: 10)

	Returns:
		Compiled StateGraph ready for invocation

	Graph Structure:
		ensure_index → retrieve → maybe_refine → generate → END

	State:
		Input:
			- question: str (user's question)
		Output:
			- contexts: List[Tuple[str, str, int, int, float]] (retrieved chunks)
			- answer: str (LLM-generated response)

	Example:
		>>> graph = build_graph(store, embedder, groq, initial_top_k=5)
		>>> result = graph.invoke({"question": "What is Python?"})
		>>> print(result["answer"])
		Python is a programming language [1]...

		Sources:
		[1] python_intro.md:1-5
	"""
	# Create StateGraph with Pydantic RagState (type-safe)
	graph = StateGraph(RagState)

	def ensure_index(state: RagState) -> Dict[str, Any]:
		"""
		Validate that vector index exists and is initialized.

		This node runs first to catch errors early if the index is empty
		or corrupted. It calls store.ensure_manifest() which creates the
		manifest file if missing.

		Args:
			state: Current RAG state

		Returns:
			Empty dict (no state updates)

		Side Effects:
			May create manifest.json if it doesn't exist
		"""
		store.ensure_manifest()
		return {}

	# Track current top_k value (starts at initial_top_k, may be increased by maybe_refine)
	current_top_k = initial_top_k

	def retrieve(state: RagState) -> Dict[str, Any]:
		"""
		Retrieve relevant document chunks using vector similarity search.

		This node:
		1. Extracts the user's question from state
		2. Encodes it into an embedding vector
		3. Searches the vector store for top-k similar chunks
		4. Formats results with line numbers (approximation)
		5. Returns contexts update

		Args:
			state: RAG state with question field

		Returns:
			Dict with "contexts" field to update state

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
		question = state.question

		# Encode the question into an embedding vector
		q_emb = embedder.encode_query(question)

		# Search vector store for top-k most similar chunks
		# Use nonlocal current_top_k which may be adjusted by maybe_refine
		nonlocal current_top_k
		results = store.search(q_emb, top_k=current_top_k)

		# Format results as (file_path, text, start_line, end_line, score) tuples
		contexts: List[Tuple[str, str, int, int, float]] = []
		for entry, score in results:
			# Note: entry.text is already the chunk text (we don't store full docs)
			# Line numbers are approximate - we count newlines within the chunk
			full_text = entry.text
			start_line = 1  # Chunks start at line 1 (relative to chunk, not document)
			end_line = full_text.count("\n") + 1

			contexts.append((entry.file_path, entry.text, start_line, end_line, score))

		# Return dict to update state
		return {"contexts": contexts}

	def maybe_refine(state: RagState) -> Dict[str, Any]:
		"""
		Expand retrieval if initial results have low similarity scores.

		This adaptive retrieval node checks if the best result has a low
		similarity score, indicating the query may not have good matches.
		If so, it increases top_k and re-runs retrieval.

		Args:
			state: RAG state with contexts from retrieve node

		Returns:
			Dict with updated "contexts" field if re-retrieval occurred, empty dict otherwise

		Adaptive Logic:
			- If max(similarity_scores) < low_score_threshold → increase top_k, re-retrieve
			- Otherwise → pass through unchanged

		Rationale:
			Low similarity means the question may be tangentially related to
			documents. Retrieving more chunks gives the LLM more context to
			work with, potentially finding a weak match.

		Example:
			Initial retrieval: top_k=5, max_score=0.28 (low!)
			→ Increase to top_k=8, re-retrieve
			→ New max_score=0.41 (better)
			→ Continue to generation with 8 chunks
		"""
		# Get current contexts from previous retrieve call
		contexts = state.contexts or []

		# Check if we have low-quality results
		# If no contexts or all scores < threshold, expand search
		if not contexts or max(c[-1] for c in contexts) < low_score_threshold:
			# Increase top_k (cap at max_top_k to avoid overwhelming the LLM)
			nonlocal current_top_k
			current_top_k = min(max_top_k, current_top_k + refinement_increase)
			# Re-run retrieval with expanded top_k
			return retrieve(state)

		# Results are good enough, continue with current contexts
		return {}

	def generate(state: RagState) -> Dict[str, Any]:
		"""
		Generate answer using LLM with retrieved context.

		This final node:
		1. Extracts question and contexts from state
		2. Formats them into LLM messages (via prompts.py)
		3. Calls Groq LLM to generate answer
		4. Returns answer update

		Args:
			state: RAG state with question and contexts fields

		Returns:
			Dict with "answer" field to update state

		LLM Interaction:
			Messages = [SystemMessage, HumanMessage with context + question]
			System prompt enforces: citation discipline, conciseness, "I don't know"
			Response should include inline citations [1], [2] and source list at end

		Example Flow:
			Input state: RagState(question="What is CurioOS?", contexts=[...])
			↓
			Build messages: [
				SystemMessage(content="You are CurioOS. Answer concisely..."),
				HumanMessage(content="Context:\\n[1] file.txt...\\n\\nQuestion: What is CurioOS?")
			]
			↓
			Call groq.generate(messages)
			↓
			Output: {"answer": "CurioOS is..."}
		"""
		question = state.question
		contexts = state.contexts or []

		# Build LLM messages with system prompt and formatted context
		# Returns list of LangChain message objects
		messages = build_messages(question, contexts)

		# Generate answer using Groq LLM
		answer = groq.generate(messages)

		# Return dict to update state
		return {"answer": answer}

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
