"""
Vector Store Module

This module implements a lightweight local vector database for CurioOS using
NumPy arrays and JSON files. It provides efficient k-nearest-neighbors search
using sklearn's NearestNeighbors with cosine similarity.

Key features:
- Local-first storage (no external database required)
- Atomic file updates (save everything on each change)
- Efficient k-NN search using sklearn
- Metadata tracking (file paths, character offsets, content hashes, timestamps)
- Change detection via MD5 hashes

Storage Format:
	embeddings.npy  - NumPy array (N x D) of float32 embeddings
	index.json      - JSON array of IndexEntry metadata
	manifest.json   - Index metadata (model name, count, dimensions, timestamp)

Design Philosophy:
	Simplicity over scalability. For small-to-medium document collections
	(<10k chunks), in-memory numpy arrays + sklearn are fast enough and
	far simpler than vector databases like Pinecone, Weaviate, or Chroma.

Typical Usage:
	>>> store = VectorStore(index_dir, "sentence-transformers/all-MiniLM-L6-v2")
	>>> store.upsert_chunks(file_path, md5, chunks, embeddings)
	>>> results = store.search(query_embedding, top_k=5)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors  # type: ignore


@dataclass
class IndexEntry:
	"""
	Metadata for a single text chunk in the vector index.

	Each chunk is a contiguous span of text from a source document,
	along with its embedding and metadata for citation/provenance.

	Attributes:
		id: Unique integer ID for this chunk (auto-incremented)
		file_path: Absolute path to source document (as string)
		chunk_start: Character offset where chunk begins in source text
		chunk_end: Character offset where chunk ends in source text
		md5: MD5 hash of the entire source document (for change detection)
		added_at: Unix timestamp when chunk was added to index
		text: The actual text content of the chunk
	"""
	id: int
	file_path: str
	chunk_start: int
	chunk_end: int
	md5: str
	added_at: float
	text: str


class VectorStore:
	"""
	Lightweight local vector store persisted to disk using NumPy and JSON.

	This class manages a collection of text chunk embeddings and their metadata.
	It provides efficient similarity search and supports incremental updates
	(adding/removing files without rebuilding entire index).

	Storage Strategy:
		All data is stored locally in the index_dir:
		- embeddings.npy: Dense embedding matrix (N chunks × D dimensions)
		- index.json: List of IndexEntry objects with metadata
		- manifest.json: Index-level metadata (model, count, dimensions, timestamp)

	Thread Safety:
		Not thread-safe. Concurrent access from multiple processes/threads
		may corrupt the index. Use file locking if needed.

	Performance:
		- Loading: O(N) - reads all embeddings into memory
		- Search: O(N log k) - sklearn's NearestNeighbors
		- Upsert: O(N) - rewrites entire index (acceptable for <10k chunks)

	Limitations:
		- Entire index must fit in RAM
		- No incremental saving (full rewrite on every change)
		- No distributed/sharded storage
	"""

	def __init__(self, index_dir: Path, embed_model_name: str):
		"""
		Initialize vector store from disk or create new empty store.

		Args:
			index_dir: Directory to store index files
			embed_model_name: Name of embedding model (stored in manifest for validation)

		Note:
			If index_dir exists and contains valid index files, they are loaded.
			Otherwise, starts with an empty index.
		"""
		self.index_dir = index_dir
		self.embed_model_name = embed_model_name

		# Paths to index files
		self.embeddings_path = self.index_dir / "embeddings.npy"
		self.index_json_path = self.index_dir / "index.json"
		self.manifest_path = self.index_dir / "manifest.json"

		# In-memory state
		self._embeddings: Optional[np.ndarray] = None  # Shape: (N, D)
		self._entries: List[IndexEntry] = []
		self._nn: Optional[NearestNeighbors] = None  # Nearest neighbors index

		# Ensure directory exists and load existing index
		self.index_dir.mkdir(parents=True, exist_ok=True)
		self._load_all()

	def _load_all(self) -> None:
		"""
		Load embeddings and metadata from disk.

		This method is called during initialization to restore the index
		from persistent storage. If files don't exist (first run), starts
		with empty arrays.

		Files Loaded:
			- embeddings.npy: NumPy array of embeddings
			- index.json: List of IndexEntry dictionaries

		Post-Condition:
			self._embeddings and self._entries are populated, and
			self._nn (nearest neighbors index) is built.
		"""
		# Load embeddings array if it exists
		if self.embeddings_path.exists():
			self._embeddings = np.load(self.embeddings_path)
		else:
			self._embeddings = None

		# Load metadata entries if they exist
		if self.index_json_path.exists():
			raw = json.loads(self.index_json_path.read_text(encoding="utf-8"))
			self._entries = [IndexEntry(**e) for e in raw]
		else:
			self._entries = []

		# Build nearest neighbors index for efficient search
		self._rebuild_nn()

	def _save_all(self) -> None:
		"""
		Persist all index data to disk atomically.

		This method writes three files:
		1. embeddings.npy: NumPy array of embeddings
		2. index.json: JSON array of IndexEntry metadata
		3. manifest.json: Index-level metadata

		The writes are not truly atomic (no transaction support), but
		each file is written completely before the next, minimizing
		the window for corruption.

		File Formats:
			- embeddings.npy: Binary NumPy format (float32)
			- index.json: Human-readable JSON with pretty-printing
			- manifest.json: Human-readable JSON with metadata
		"""
		# Save embeddings array (or delete if empty)
		if self._embeddings is not None:
			np.save(self.embeddings_path, self._embeddings.astype(np.float32, copy=False))
		else:
			# Remove embeddings file if index is empty
			if self.embeddings_path.exists():
				self.embeddings_path.unlink(missing_ok=True)

		# Save metadata entries as JSON
		self.index_json_path.write_text(
			json.dumps([asdict(e) for e in self._entries], ensure_ascii=False, indent=2),
			encoding="utf-8"
		)

		# Save manifest with index-level metadata
		manifest = {
			"embed_model": self.embed_model_name,
			"count": len(self._entries),
			"dim": None if self._embeddings is None else int(self._embeddings.shape[1]),
			"updated_at": time.time(),
		}
		self.manifest_path.write_text(
			json.dumps(manifest, ensure_ascii=False, indent=2),
			encoding="utf-8"
		)

	def _rebuild_nn(self) -> None:
		"""
		Rebuild the nearest neighbors index for fast similarity search.

		This method creates a new NearestNeighbors object from sklearn
		and fits it on the current embeddings. Must be called after any
		change to self._embeddings.

		Algorithm:
			sklearn's NearestNeighbors with cosine metric
			- Exact search (not approximate)
			- O(N log k) query time
			- O(N) build time

		Note:
			If index is empty, sets self._nn to None (no search possible).
		"""
		if self._embeddings is None or len(self._entries) == 0:
			self._nn = None
			return

		# Build nearest neighbors index with cosine similarity
		# Cosine metric is appropriate for normalized embeddings
		self._nn = NearestNeighbors(metric="cosine")
		self._nn.fit(self._embeddings)

	def ensure_manifest(self) -> None:
		"""
		Ensure manifest file exists, creating it if necessary.

		This is called at application startup to verify the index is
		initialized. If manifest doesn't exist, triggers a full save.
		"""
		if not self.manifest_path.exists():
			self._save_all()

	def upsert_chunks(self, file_path: Path, md5: str, chunks: List[Tuple[int, int, str]], embeddings: np.ndarray) -> None:
		"""
		Replace all chunks for a given file with new ones.

		This implements an "upsert" (update or insert) operation:
		1. Remove all existing chunks for this file
		2. Add new chunks with fresh embeddings
		3. Save index to disk
		4. Rebuild nearest neighbors index

		Args:
			file_path: Path to source document
			md5: MD5 hash of current document content
			chunks: List of (start_char, end_char, text) tuples
			embeddings: NumPy array of embeddings (same length as chunks)

		Example:
			>>> chunks = [(0, 100, "First chunk"), (100, 200, "Second chunk")]
			>>> embeddings = embedder.encode_texts([c[2] for c in chunks])
			>>> store.upsert_chunks(Path("doc.txt"), "abc123", chunks, embeddings)

		Performance:
			O(N) where N = total chunks in index
			- Must remove old chunks (linear scan)
			- Must rebuild entire numpy array (vstack)
			- Must rebuild nearest neighbors index
		"""
		# Step 1: Remove old entries for this file (if any)
		old_ids = [e.id for e in self._entries if Path(e.file_path) == file_path]
		if old_ids:
			self._remove_ids(old_ids)

		# Step 2: Create new entries with auto-incremented IDs
		start_id = 0 if not self._entries else max(e.id for e in self._entries) + 1
		new_entries: List[IndexEntry] = []
		for i, (span, emb) in enumerate(zip(chunks, embeddings)):
			ch_start, ch_end, ch_text = span
			new_entries.append(
				IndexEntry(
					id=start_id + i,
					file_path=str(file_path),
					chunk_start=ch_start,
					chunk_end=ch_end,
					md5=md5,
					added_at=time.time(),
					text=ch_text,
				)
			)

		# Step 3: Append new data to index
		if self._embeddings is None or len(self._entries) == 0:
			# First chunks in the index
			self._embeddings = embeddings.copy()
			self._entries = new_entries
		else:
			# Append to existing index
			self._embeddings = np.vstack([self._embeddings, embeddings])
			self._entries.extend(new_entries)

		# Step 4: Persist and rebuild search index
		self._save_all()
		self._rebuild_nn()

	def _remove_ids(self, ids: List[int]) -> None:
		"""
		Remove chunks with given IDs from the index (in-memory only).

		This is a helper method used by upsert_chunks and remove_file.
		It does NOT save to disk or rebuild the NN index - caller must do that.

		Args:
			ids: List of chunk IDs to remove

		Implementation:
			Uses boolean masking to filter both embeddings array and entries list.
		"""
		if not ids:
			return

		# Create boolean mask: True if entry should be kept
		keep_mask = [e.id not in ids for e in self._entries]

		# Filter embeddings array
		if self._embeddings is not None:
			keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
			if keep_indices:
				self._embeddings = self._embeddings[keep_indices, :]
			else:
				# All chunks removed - reset to None
				self._embeddings = None

		# Filter entries list
		self._entries = [e for e in self._entries if e.id not in ids]

	def remove_file(self, file_path: Path) -> None:
		"""
		Remove all chunks associated with a given file from the index.

		This is called when a file is deleted from the vault. It:
		1. Finds all chunks belonging to the file
		2. Removes them from the index
		3. Saves updated index to disk
		4. Rebuilds nearest neighbors index

		Args:
			file_path: Path to file whose chunks should be removed

		Example:
			>>> store.remove_file(Path("data/vault/deleted_doc.txt"))
		"""
		# Find all chunk IDs for this file
		rm_ids = [e.id for e in self._entries if Path(e.file_path) == file_path]

		# Remove them from the index
		self._remove_ids(rm_ids)

		# Persist changes and rebuild search index
		self._save_all()
		self._rebuild_nn()

	def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[IndexEntry, float]]:
		"""
		Find the top-k most similar chunks to a query embedding.

		This is the core search operation for RAG. Given a query embedding,
		it returns the most semantically similar document chunks along with
		their similarity scores.

		Args:
			query_embedding: Query embedding vector (shape: (D,))
			top_k: Number of results to return (default: 5)

		Returns:
			List of (IndexEntry, similarity_score) tuples, sorted by score (best first)
			- similarity_score: Float in [0, 1], where 1 = identical, 0 = orthogonal
			- IndexEntry contains chunk text, file path, and character offsets

		Similarity Metric:
			Cosine similarity, converted from cosine distance:
			- Cosine distance = 1 - cosine similarity
			- Returned score = 1 - distance

		Edge Cases:
			- If index is empty, returns []
			- If top_k > index size, returns all chunks

		Example:
			>>> query_emb = embedder.encode_query("What is CurioOS?")
			>>> results = store.search(query_emb, top_k=5)
			>>> for entry, score in results:
			...     print(f"{score:.2f}: {entry.file_path}:{entry.chunk_start}-{entry.chunk_end}")
		"""
		# Handle empty index
		if self._nn is None or self._embeddings is None or len(self._entries) == 0:
			return []

		# Clamp top_k to index size
		k = min(top_k, len(self._entries))

		# Perform k-NN search using sklearn
		# Returns: distances (cosine distance), indices (row indices in embeddings array)
		distances, indices = self._nn.kneighbors([query_embedding], n_neighbors=k, return_distance=True)

		# Convert results to (IndexEntry, similarity_score) tuples
		results: List[Tuple[IndexEntry, float]] = []
		for dist, idx in zip(distances[0], indices[0]):
			entry = self._entries[int(idx)]
			# Convert cosine distance to cosine similarity
			# Cosine distance = 1 - cosine similarity (for unit vectors)
			score = 1.0 - float(dist)
			results.append((entry, score))

		return results
