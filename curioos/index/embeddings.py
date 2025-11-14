from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore


class Embedder:
	"""Wrapper around SentenceTransformers to provide consistent numpy outputs."""

	def __init__(self, model_name: str, cache_folder: Optional[Path] = None):
		self.model_name = model_name
		self.cache_folder = cache_folder

		# Set environment variable for sentence-transformers cache
		if cache_folder:
			os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_folder)

		self._model: SentenceTransformer = SentenceTransformer(
			model_name,
			cache_folder=str(cache_folder) if cache_folder else None
		)

	def encode_texts(self, texts: List[str]) -> np.ndarray:
		emb = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
		return emb.astype(np.float32, copy=False)

	def encode_query(self, text: str) -> np.ndarray:
		return self.encode_texts([text])[0]


