from __future__ import annotations

from typing import List, Tuple


SYSTEM_PROMPT = (
	"You are CurioOS. Answer concisely using only the provided context. "
	"If the answer is not in the context, say you don't know. "
	"Cite sources inline as [n] and list them at the end as file:line-range."
)


def format_context(contexts: List[Tuple[str, str, int, int, float]]) -> str:
	"""Build the context string from (file_path, text, start_line, end_line, score)."""
	parts = []
	for i, (file_path, text, start_line, end_line, score) in enumerate(contexts, start=1):
		header = f"[{i}] {file_path}:{start_line}-{end_line} (score={score:.2f})"
		parts.append(f"{header}\n{text}")
	return "\n\n---\n\n".join(parts)


def build_messages(question: str, contexts: List[Tuple[str, str, int, int, float]]) -> list:
	context_str = format_context(contexts) if contexts else "No context available."
	user = f"Context:\n{context_str}\n\nQuestion: {question}"
	return [
		{"role": "system", "content": SYSTEM_PROMPT},
		{"role": "user", "content": user},
	]


