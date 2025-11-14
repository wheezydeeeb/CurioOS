from __future__ import annotations

from typing import Iterable, List, Tuple


def _paragraphs(text: str) -> List[str]:
	parts: List[str] = []
	buf: List[str] = []
	for line in text.split("\n"):
		if line.strip() == "":
			if buf:
				parts.append("\n".join(buf).strip())
				buf = []
		else:
			buf.append(line)
	if buf:
		parts.append("\n".join(buf).strip())
	return parts


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[Tuple[int, int, str]]:
	"""Split text into overlapping chunks; returns (start_char, end_char, chunk_text)."""
	if chunk_size <= 0 or overlap < 0:
		raise ValueError("chunk_size must be > 0 and overlap must be >= 0")

	paras = _paragraphs(text)
	offsets: List[Tuple[int, int]] = []
	cursor = 0
	for p in paras:
		start = text.find(p, cursor)
		if start == -1:
			start = cursor
		end = start + len(p)
		offsets.append((start, end))
		cursor = end

	chunks: List[Tuple[int, int, str]] = []
	build: List[Tuple[int, int, str]] = []
	buf = ""
	buf_start = 0
	for (start, end), p in zip(offsets, paras):
		if not buf:
			buf_start = start
		added = p if not buf else f"{buf}\n\n{p}"
		if len(added) <= chunk_size:
			buf = added
		else:
			if buf:
				chunks.append((buf_start, buf_start + len(buf), buf))
				# overlap window
				ov_text = buf[-overlap:] if overlap > 0 else ""
				buf = ov_text
				buf_start = buf_start + (len(buf) - len(ov_text))
			if len(p) > chunk_size:
				# hard-split long paragraph
				i = 0
				while i < len(p):
					s = i
					e = min(i + chunk_size, len(p))
					sub = p[s:e]
					ch_start = start + s
					ch_end = start + e
					chunks.append((ch_start, ch_end, sub))
					i += chunk_size - overlap if overlap < chunk_size else chunk_size
				buf = ""
			else:
				buf = p
				buf_start = start

	if buf:
		chunks.append((buf_start, buf_start + len(buf), buf))

	return chunks


