"""
RAG Prompt Engineering Module

This module contains all prompt templates and message formatting logic for CurioOS's
RAG pipeline. Good prompts are critical for RAG quality - they instruct the LLM on:
- How to use retrieved context
- When to say "I don't know"
- How to cite sources
- Output format and tone

Key components:
- SYSTEM_PROMPT: Core instruction that defines CurioOS's behavior
- format_context(): Converts search results into numbered context blocks
- build_messages(): Assembles system + user message for LLM

Design Philosophy:
	Prompt engineering for RAG is a delicate balance:
	1. Enforce citation discipline (prevents hallucination)
	2. Allow "I don't know" (prevents making up answers)
	3. Encourage conciseness (faster, cheaper responses)
	4. Format context clearly (helps LLM find relevant info)

Typical Usage:
	>>> contexts = [(file_path, text, start_line, end_line, score), ...]
	>>> messages = build_messages("What is CurioOS?", contexts)
	>>> answer = llm.generate(messages)
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.messages import SystemMessage, HumanMessage  # type: ignore


# System prompt that defines CurioOS's personality and constraints
# This is sent with EVERY request to set behavioral expectations
SYSTEM_PROMPT = (
	"You are CurioOS. Answer concisely using only the provided context. "
	"If the answer is not in the context, say you don't know. "
	"Cite sources inline as [n] and list them at the end as file:line-range."
)


def format_context(contexts: List[Tuple[str, str, int, int, float]]) -> str:
	"""
	Format retrieved context chunks into a structured string for the LLM.

	Each context chunk is formatted with:
	- Numbered citation tag [1], [2], etc.
	- Source file path and line range
	- Similarity score (helps LLM prioritize relevant chunks)
	- The actual text content

	Args:
		contexts: List of (file_path, text, start_line, end_line, score) tuples
		          Typically from VectorStore.search() results

	Returns:
		Formatted context string ready for LLM consumption

	Format:
		[1] path/to/file.txt:5-12 (score=0.87)
		<chunk text here>

		---

		[2] path/to/file2.md:20-25 (score=0.82)
		<chunk text here>

	Example:
		>>> contexts = [
		...     ("notes.txt", "CurioOS is a RAG system", 1, 1, 0.89),
		...     ("readme.md", "Built with Python", 5, 5, 0.75)
		... ]
		>>> print(format_context(contexts))
		[1] notes.txt:1-1 (score=0.89)
		CurioOS is a RAG system

		---

		[2] readme.md:5-5 (score=0.75)
		Built with Python

	Design Rationale:
		- Numbered tags [n] enable inline citation in answers
		- File paths provide provenance for verification
		- Line ranges help users locate exact source location
		- Scores help LLM weigh evidence (higher score = more relevant)
		- "---" separators clearly delineate chunks
	"""
	parts = []
	for i, (file_path, text, start_line, end_line, score) in enumerate(contexts, start=1):
		# Format header with citation number, source, and similarity score
		header = f"[{i}] {file_path}:{start_line}-{end_line} (score={score:.2f})"
		# Combine header and text content
		parts.append(f"{header}\n{text}")

	# Join all chunks with separator
	return "\n\n---\n\n".join(parts)


def build_messages(question: str, contexts: List[Tuple[str, str, int, int, float]]) -> List[SystemMessage | HumanMessage]:
	"""
	Build the complete message list for LLM generation.

	This function assembles a two-message conversation:
	1. System message: Defines CurioOS's behavior and constraints
	2. Human message: Contains formatted context + user's question

	Args:
		question: User's question (e.g., "What is CurioOS?")
		contexts: Retrieved context chunks from vector search

	Returns:
		List of LangChain message objects ready for GroqClient.generate()
		Format: [SystemMessage(...), HumanMessage(...)]

	Message Structure:
		SystemMessage:
			"You are CurioOS. Answer concisely using only the provided context..."

		HumanMessage:
			"Context:
			[1] file1.txt:5-10 (score=0.87)
			<chunk text>
			---
			[2] file2.md:20-25 (score=0.82)
			<chunk text>

			Question: What is CurioOS?"

	Example:
		>>> contexts = [("doc.txt", "CurioOS is awesome", 1, 1, 0.9)]
		>>> messages = build_messages("What is CurioOS?", contexts)
		>>> print(messages)
		[
		  SystemMessage(content="You are CurioOS. Answer concisely..."),
		  HumanMessage(content="Context:\\n[1] doc.txt:1-1 (score=0.90)\\nCurioOS is awesome\\n\\nQuestion: What is CurioOS?")
		]

	Design Rationale:
		- System message ensures consistent behavior across all queries
		- Context placed before question (LLM sees evidence before forming answer)
		- Clear separation between context and question
		- "No context available" fallback prevents errors if search returns nothing
		- Using LangChain message types provides better type safety and integration
	"""
	# Format context chunks, or use placeholder if no results
	context_str = format_context(contexts) if contexts else "No context available."

	# Build user message with context first, then question
	# This ordering helps the LLM ground its answer in the context
	user_content = f"Context:\n{context_str}\n\nQuestion: {question}"

	# Return standard message list using LangChain message types
	return [
		SystemMessage(content=SYSTEM_PROMPT),
		HumanMessage(content=user_content),
	]
