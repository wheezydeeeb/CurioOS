"""
Text Embedding Module

This module provides a wrapper around Sentence Transformers for generating
dense vector embeddings of text. Embeddings are numerical representations
that capture semantic meaning, enabling similarity search.

Key features:
- Consistent numpy output format (float32, normalized)
- Local model caching (no repeated downloads)
- Batch encoding for efficiency
- Single-query convenience method

Model Information:
	Default model: sentence-transformers/all-MiniLM-L6-v2
	- Dimensions: 384
	- Size: ~90MB
	- Speed: ~14,000 sentences/sec on CPU
	- Quality: Good for general-purpose semantic search

Design Philosophy:
	All embeddings are L2-normalized (unit vectors), so cosine similarity
	can be computed as simple dot product. This makes vector search more
	efficient and numerically stable.

Typical Usage:
	>>> embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2", cache_folder)
	>>> # Batch encode for indexing
	>>> embeddings = embedder.encode_texts(["chunk1", "chunk2", "chunk3"])
	>>> # Single encode for queries
	>>> query_embedding = embedder.encode_query("What is CurioOS?")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore


class Embedder:
	"""
	Wrapper around SentenceTransformers for consistent embedding generation.

	This class provides a clean interface to the sentence-transformers library
	with opinionated defaults for CurioOS's use case. It ensures all embeddings
	are normalized and returned as float32 numpy arrays.
	"""

	def __init__(self, model_name: str, cache_folder: Optional[Path] = None):
		"""
		Initialize the embedder with a specific model.

		Args:
			model_name: Hugging Face model identifier or local path
			            Examples:
			            - "sentence-transformers/all-MiniLM-L6-v2" (default, 384 dims)
			            - "sentence-transformers/all-mpnet-base-v2" (768 dims, slower but higher quality)
			            - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (multilingual)
			cache_folder: Optional directory to cache model files locally
			              If None, uses default sentence-transformers cache location
			              Setting this allows offline operation after first download

		Model Download:
			On first use, the model is downloaded from Hugging Face (unless already cached).
			Subsequent uses load from cache. Model files include:
			- PyTorch weights
			- Tokenizer vocabulary
			- Configuration files

		Note:
			Model loading can take a few seconds on first call. For production,
			consider pre-downloading models during setup.
		"""
		self.model_name = model_name
		self.cache_folder = cache_folder

		# Configure sentence-transformers to use our cache directory
		# This environment variable controls where models are stored
		if cache_folder:
			os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_folder)

		# Load the model (downloads if not cached)
		# This is blocking and can take several seconds on first run
		self._model: SentenceTransformer = SentenceTransformer(
			model_name,
			cache_folder=str(cache_folder) if cache_folder else None
		)

	def encode_texts(self, texts: List[str]) -> np.ndarray:
		"""
		Encode a batch of texts into embeddings.

		This method is optimized for batch processing and should be used
		when encoding multiple texts (e.g., during indexing). Batch encoding
		is significantly faster than encoding texts one-by-one.

		Args:
			texts: List of text strings to encode
			       Can be sentences, paragraphs, or entire documents
			       (subject to model's max sequence length, typically 512 tokens)

		Returns:
			NumPy array of shape (len(texts), embedding_dim)
			- dtype: float32 (optimized for storage and computation)
			- Each row is an L2-normalized embedding vector
			- Example shape: (100, 384) for 100 texts with all-MiniLM-L6-v2

		Normalization:
			All vectors are L2-normalized (unit length), which means:
			- ||embedding|| = 1.0 for all embeddings
			- Cosine similarity = dot product
			- This is standard practice for semantic search

		Performance:
			Encoding speed depends on:
			- Text length (longer texts take more time)
			- Batch size (larger batches are more efficient)
			- Hardware (GPU >> CPU, but CPU is sufficient for small datasets)

		Example:
			>>> texts = ["Hello world", "Goodbye world", "CurioOS is awesome"]
			>>> embeddings = embedder.encode_texts(texts)
			>>> print(embeddings.shape)
			(3, 384)
			>>> print(np.linalg.norm(embeddings[0]))  # Check normalization
			1.0
		"""
		# Call sentence-transformers encode with our preferred settings:
		# - show_progress_bar=False: No progress output (cleaner logs)
		# - convert_to_numpy=True: Return numpy array instead of torch tensor
		# - normalize_embeddings=True: L2-normalize all vectors
		emb = self._model.encode(
			texts,
			show_progress_bar=False,
			convert_to_numpy=True,
			normalize_embeddings=True
		)

		# Ensure float32 dtype (some models might return float64)
		# copy=False means no-op if already float32
		return emb.astype(np.float32, copy=False)

	def encode_query(self, text: str) -> np.ndarray:
		"""
		Encode a single text string into an embedding vector.

		This is a convenience method for encoding individual queries.
		Internally, it calls encode_texts with a single-element list
		and extracts the first result.

		Args:
			text: Query text to encode (typically a user question)

		Returns:
			NumPy array of shape (embedding_dim,)
			- dtype: float32
			- L2-normalized (unit length)
			- Example shape: (384,) for all-MiniLM-L6-v2

		Example:
			>>> query_emb = embedder.encode_query("What is CurioOS?")
			>>> print(query_emb.shape)
			(384,)
			>>> # Can now compare to document embeddings via dot product
			>>> scores = doc_embeddings @ query_emb
		"""
		# Encode as single-element batch and extract first result
		return self.encode_texts([text])[0]
