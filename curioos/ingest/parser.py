from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple

from pdfminer.high_level import extract_text  # type: ignore


def _read_text_file(path: Path) -> str:
	text = path.read_text(encoding="utf-8", errors="ignore")
	return text


def _read_pdf_file(path: Path) -> str:
	try:
		text = extract_text(str(path)) or ""
	except Exception:
		text = ""
	return text


def normalize_text(text: str) -> str:
	# Standardize newlines
	text = text.replace("\r\n", "\n").replace("\r", "\n")
	# Collapse excessive blank lines
	lines = [ln.rstrip() for ln in text.split("\n")]
	normalized = []
	blank = 0
	for ln in lines:
		if ln.strip() == "":
			blank += 1
			if blank <= 1:
				normalized.append("")
		else:
			blank = 0
			normalized.append(ln)
	return "\n".join(normalized).strip()


def md5_of_text(text: str) -> str:
	h = hashlib.md5()
	h.update(text.encode("utf-8", errors="ignore"))
	return h.hexdigest()


def load_and_normalize(path: Path) -> Tuple[str, str]:
	"""Return (normalized_text, md5) for allowed file types."""
	ext = path.suffix.lower()
	if ext in {".txt", ".md"}:
		raw = _read_text_file(path)
	elif ext == ".pdf":
		raw = _read_pdf_file(path)
	else:
		raise ValueError(f"Unsupported file type: {ext}")
	text = normalize_text(raw)
	return text, md5_of_text(text)


