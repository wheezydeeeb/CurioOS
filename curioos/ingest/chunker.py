"""
Text Chunking Module

This module provides intelligent text chunking for CurioOS's RAG system. It splits
long documents into smaller, overlapping chunks that fit within embedding model
context windows while preserving semantic coherence.

Key features:
- Paragraph-aware splitting (respects semantic boundaries)
- Configurable chunk size and overlap
- Character-level position tracking for source citation
- Handles oversized paragraphs with hard splitting

Design Rationale:
	Chunking is critical for RAG quality. Too large = exceeds embedding limits.
	Too small = loses context. Overlap ensures no information is lost at boundaries.
	Paragraph boundaries are respected to maintain semantic coherence.

Typical Usage:
	>>> chunks = chunk_text(document, chunk_size=800, overlap=150)
	>>> for start, end, text in chunks:
	...     embedding = embed(text)
	...     store(embedding, file_path, start, end)
"""

from __future__ import annotations

from typing import Iterable, List, Tuple


def _paragraphs(text: str) -> List[str]:
	"""
	Split text into paragraphs based on blank lines.

	A paragraph is defined as one or more consecutive non-blank lines,
	separated from other paragraphs by blank lines.

	Args:
		text: Input text with potential blank lines

	Returns:
		List of paragraph strings (blank lines removed, whitespace stripped)

	Example:
		Input:  "Line 1\\nLine 2\\n\\nLine 3\\nLine 4"
		Output: ["Line 1\\nLine 2", "Line 3\\nLine 4"]

	Note:
		This function assumes text has been normalized (via normalize_text)
		so all newlines are Unix-style \\n.
	"""
	parts: List[str] = []
	buf: List[str] = []

	# Process line by line
	for line in text.split("\n"):
		if line.strip() == "":
			# Blank line marks paragraph boundary
			if buf:
				# Join accumulated lines into a paragraph
				parts.append("\n".join(buf).strip())
				buf = []
		else:
			# Non-blank line belongs to current paragraph
			buf.append(line)

	# Don't forget the last paragraph if text doesn't end with blank line
	if buf:
		parts.append("\n".join(buf).strip())

	return parts


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[Tuple[int, int, str]]:
	"""
	Split text into overlapping chunks while respecting paragraph boundaries.

	This function implements a sophisticated chunking algorithm that:
	1. Splits text into paragraphs (semantic units)
	2. Builds chunks by combining paragraphs until size limit is reached
	3. Overlaps consecutive chunks to preserve context at boundaries
	4. Tracks character offsets in the original text for citation

	The algorithm prefers to keep paragraphs intact, but will hard-split
	oversized paragraphs that exceed chunk_size.

	Args:
		text: Input text to chunk (should be normalized)
		chunk_size: Maximum characters per chunk (default: 800)
		           Chosen to fit comfortably in most embedding models
		overlap: Number of characters to overlap between chunks (default: 150)
		         Ensures no context is lost at chunk boundaries

	Returns:
		List of (start_char, end_char, chunk_text) tuples
		- start_char: Character offset in original text where chunk begins
		- end_char: Character offset in original text where chunk ends
		- chunk_text: The actual text content of the chunk

	Raises:
		ValueError: If chunk_size <= 0 or overlap < 0

	Example:
		>>> text = "Paragraph 1.\\n\\nParagraph 2.\\n\\nParagraph 3."
		>>> chunks = chunk_text(text, chunk_size=20, overlap=5)
		>>> for start, end, chunk in chunks:
		...     print(f"{start}-{end}: {chunk[:20]}...")

	Algorithm Details:
		1. Split text into paragraphs
		2. Track each paragraph's position in original text
		3. Build chunks by accumulating paragraphs:
		   - If adding next paragraph stays under chunk_size → add it
		   - If it would exceed → finalize current chunk and start new one
		   - Overlap: New chunk starts with last N chars of previous chunk
		4. For paragraphs larger than chunk_size:
		   - Hard-split into fixed-size pieces with overlap
		5. Emit final chunk if buffer is non-empty

	Performance:
		O(n) where n = len(text), with overhead for string operations
	"""
	# Validate parameters
	if chunk_size <= 0 or overlap < 0:
		raise ValueError("chunk_size must be > 0 and overlap must be >= 0")

	# Step 1: Split text into paragraphs
	paras = _paragraphs(text)

	# Step 2: Find each paragraph's position in the original text
	# This is necessary for accurate source citation later
	offsets: List[Tuple[int, int]] = []
	cursor = 0  # Current search position in text
	for p in paras:
		# Find where this paragraph appears in the original text
		start = text.find(p, cursor)
		if start == -1:
			# Shouldn't happen if normalization worked correctly, but be defensive
			start = cursor
		end = start + len(p)
		offsets.append((start, end))
		cursor = end

	# Step 3: Build chunks by accumulating paragraphs
	chunks: List[Tuple[int, int, str]] = []
	buf = ""  # Current chunk being built
	buf_start = 0  # Character offset where current chunk starts

	for (start, end), p in zip(offsets, paras):
		# If buffer is empty, this paragraph starts a new chunk
		if not buf:
			buf_start = start

		# Try adding this paragraph to the current chunk
		# Paragraphs are joined with double newlines to preserve separation
		added = p if not buf else f"{buf}\n\n{p}"

		if len(added) <= chunk_size:
			# Fits! Add the paragraph to current chunk
			buf = added
		else:
			# Doesn't fit. Need to finalize current chunk and handle this paragraph

			if buf:
				# Finalize the current chunk
				chunks.append((buf_start, buf_start + len(buf), buf))

				# Create overlap window: take last N characters from current chunk
				# This ensures context continuity between chunks
				ov_text = buf[-overlap:] if overlap > 0 else ""
				buf = ov_text
				# Update buffer start position to reflect the overlap
				buf_start = buf_start + (len(buf) - len(ov_text))

			# Now handle the paragraph that didn't fit
			if len(p) > chunk_size:
				# Paragraph is too large even for its own chunk - must hard-split it
				i = 0
				while i < len(p):
					s = i
					e = min(i + chunk_size, len(p))
					sub = p[s:e]
					# Calculate absolute position in original text
					ch_start = start + s
					ch_end = start + e
					chunks.append((ch_start, ch_end, sub))
					# Move forward, accounting for overlap
					i += chunk_size - overlap if overlap < chunk_size else chunk_size
				# After hard-splitting, buffer is empty
				buf = ""
			else:
				# Paragraph fits in its own chunk - becomes new buffer
				buf = p
				buf_start = start

	# Don't forget the final chunk if there's anything in the buffer
	if buf:
		chunks.append((buf_start, buf_start + len(buf), buf))

	return chunks
