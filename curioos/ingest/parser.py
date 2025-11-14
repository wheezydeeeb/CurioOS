"""
Document Parser Module

This module handles reading and parsing various document formats for CurioOS.
It provides unified text extraction from different file types (.txt, .md, .pdf)
and normalizes the output for consistent downstream processing.

Key responsibilities:
- Extract raw text from supported file formats
- Normalize text (standardize newlines, collapse excessive blank lines)
- Generate content fingerprints (MD5) for change detection
- Handle encoding errors gracefully

Supported Formats:
- Plain text files (.txt)
- Markdown files (.md)
- PDF files (.pdf) - text-based only, no OCR

Design Philosophy:
	All files are converted to normalized plain text, stripping formatting
	but preserving paragraph structure. This ensures consistent chunking
	and embedding regardless of source format.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

from pdfminer.high_level import extract_text  # type: ignore


def _read_text_file(path: Path) -> str:
	"""
	Read a plain text or markdown file with UTF-8 encoding.

	Args:
		path: Path to the .txt or .md file

	Returns:
		Raw text content as string

	Note:
		Uses errors="ignore" to skip malformed UTF-8 sequences rather than
		crashing. This trades perfect accuracy for robustness.
	"""
	text = path.read_text(encoding="utf-8", errors="ignore")
	return text


def _read_pdf_file(path: Path) -> str:
	"""
	Extract text content from a PDF file using pdfminer.six.

	This function handles text-based PDFs only. Scanned PDFs (images of text)
	will return empty strings unless they have been OCR'd.

	Args:
		path: Path to the .pdf file

	Returns:
		Extracted text content, or empty string if extraction fails

	Note:
		Catches all exceptions during PDF parsing to prevent crashes from
		malformed or encrypted PDFs. Returns empty string on failure.
	"""
	try:
		# pdfminer.six can return None if PDF has no extractable text
		text = extract_text(str(path)) or ""
	except Exception:
		# Broad exception catch for corrupted PDFs, encrypted PDFs, etc.
		# Better to skip a problematic file than crash the entire indexing process
		text = ""
	return text


def normalize_text(text: str) -> str:
	"""
	Normalize text by standardizing newlines and collapsing excessive blank lines.

	This function ensures consistent text formatting regardless of source:
	1. Converts all line endings (CRLF, CR) to Unix-style (LF)
	2. Strips trailing whitespace from each line
	3. Collapses runs of multiple blank lines into at most one blank line
	4. Strips leading/trailing whitespace from the entire document

	Args:
		text: Raw text from any source

	Returns:
		Normalized text with consistent formatting

	Example:
		Input:  "Line 1\\r\\nLine 2  \\n\\n\\n\\nLine 3"
		Output: "Line 1\\nLine 2\\n\\nLine 3"

	Rationale:
		Normalization prevents duplicate embeddings for semantically identical
		content that differs only in whitespace. It also improves chunking by
		providing consistent paragraph boundaries.
	"""
	# Standardize all line endings to Unix-style \n
	# Handles files from Windows (\r\n), old Mac (\r), and Unix (\n)
	text = text.replace("\r\n", "\n").replace("\r", "\n")

	# Strip trailing whitespace from each line (but preserve leading spaces for indentation)
	lines = [ln.rstrip() for ln in text.split("\n")]

	# Collapse excessive blank lines: allow at most one consecutive blank line
	normalized = []
	blank = 0  # Counter for consecutive blank lines
	for ln in lines:
		if ln.strip() == "":
			# This is a blank line
			blank += 1
			if blank <= 1:
				# First blank line in a sequence - keep it
				normalized.append("")
		else:
			# Non-blank line - reset counter and add the line
			blank = 0
			normalized.append(ln)

	# Join lines back together and strip leading/trailing whitespace from entire document
	return "\n".join(normalized).strip()


def md5_of_text(text: str) -> str:
	"""
	Generate MD5 hash of text content for change detection.

	The MD5 fingerprint allows efficient detection of document changes without
	storing full text. When a file is modified, its hash changes, triggering
	re-indexing.

	Args:
		text: Text content to hash (should be normalized first)

	Returns:
		32-character hexadecimal MD5 hash

	Note:
		MD5 is cryptographically weak but sufficient for change detection.
		We're not using it for security, just for quickly detecting when
		content has been modified.
	"""
	h = hashlib.md5()
	h.update(text.encode("utf-8", errors="ignore"))
	return h.hexdigest()


def load_and_normalize(path: Path) -> Tuple[str, str]:
	"""
	Load a document file, extract text, normalize it, and compute content hash.

	This is the primary entry point for document ingestion. It orchestrates
	the entire parsing pipeline:
	1. Determine file type from extension
	2. Extract raw text using appropriate parser
	3. Normalize text format
	4. Compute MD5 hash for change detection

	Args:
		path: Path to document file

	Returns:
		Tuple of (normalized_text, md5_hash)

	Raises:
		ValueError: If file extension is not supported

	Supported Extensions:
		.txt  - Plain text
		.md   - Markdown (treated as plain text, formatting preserved)
		.pdf  - Portable Document Format (text extraction only)

	Example:
		>>> text, md5 = load_and_normalize(Path("./data/vault/notes.md"))
		>>> print(len(text), md5)
		4523 3f2a7b9c1d4e5f6a7b8c9d0e1f2a3b4c
	"""
	# Determine file type from extension (case-insensitive)
	ext = path.suffix.lower()

	# Route to appropriate parser based on file extension
	if ext in {".txt", ".md"}:
		raw = _read_text_file(path)
	elif ext == ".pdf":
		raw = _read_pdf_file(path)
	else:
		raise ValueError(f"Unsupported file type: {ext}")

	# Normalize the raw text and compute content hash
	text = normalize_text(raw)
	return text, md5_of_text(text)


