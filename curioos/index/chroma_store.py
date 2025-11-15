"""
ChromaDB Vector Store Module

This module implements a vector database for CurioOS using ChromaDB,
replacing the previous NumPy+JSON implementation with a proper embedded
vector database.

Key features:
- Persistent local storage via ChromaDB
- Automatic saving (no manual persistence needed)
- Efficient k-NN search with built-in indexing
- Native metadata support (file paths, character offsets, content hashes, timestamps)
- Advanced querying capabilities (metadata filtering, where clauses)

Storage Format:
	data/chroma/
	├── chroma.sqlite3          # Metadata + configuration
	└── [internal parquet files] # Embeddings storage

Design Philosophy:
	Use a proper vector database for better scalability, maintainability,
	and query capabilities while maintaining the same simple API as the
	previous implementation.

Typical Usage:
	>>> store = ChromaVectorStore(index_dir, "sentence-transformers/all-MiniLM-L6-v2")
	>>> store.upsert_chunks(file_path, md5, chunks, embeddings)
	>>> results = store.search(query_embedding, top_k=5)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.config import Settings
import numpy as np
from pydantic import BaseModel, Field


class IndexEntry(BaseModel):
	"""
	Metadata for a single text chunk in the vector index.

	This maintains compatibility with the previous VectorStore implementation
	and the RAG pipeline expectations.

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
	chunk_start: int = Field(ge=0)
	chunk_end: int = Field(ge=0)
	md5: str
	added_at: float = Field(ge=0)
	text: str


class ChromaVectorStore:
	"""
	ChromaDB-based vector store for document embeddings and metadata.

	This class provides the same interface as the previous VectorStore class
	but uses ChromaDB for persistent storage and efficient querying.

	Storage Strategy:
		All data is stored locally in the index_dir using ChromaDB's
		persistent client, which manages embeddings and metadata in
		an optimized format (SQLite + Parquet).

	Thread Safety:
		ChromaDB handles concurrent access internally. Safe for multiple
		read operations, but write operations should be serialized.

	Performance:
		- Loading: Fast (ChromaDB lazy-loads data as needed)
		- Search: O(log N) with HNSW indexing (approximate nearest neighbors)
		- Upsert: O(k) where k = number of chunks being added

	Advantages over NumPy+JSON:
		- No manual save/load logic needed
		- Better query performance for larger datasets
		- Native metadata filtering support
		- Automatic optimization and indexing
	"""

	COLLECTION_NAME = "curioos_docs"

	def __init__(self, index_dir: Path, embed_model_name: str):
		"""
		Initialize ChromaDB vector store from disk or create new collection.

		Args:
			index_dir: Directory to store ChromaDB files
			embed_model_name: Name of embedding model (stored in collection metadata)

		Note:
			If index_dir exists and contains a ChromaDB database, it is loaded.
			Otherwise, creates a new collection.
		"""
		self.index_dir = index_dir
		self.embed_model_name = embed_model_name

		# Ensure directory exists
		self.index_dir.mkdir(parents=True, exist_ok=True)

		# Initialize ChromaDB persistent client
		self._client = chromadb.PersistentClient(
			path=str(self.index_dir),
			settings=Settings(
				anonymized_telemetry=False,  # Disable telemetry for privacy
				allow_reset=True,  # Allow reset for testing/migration
			)
		)

		# Get or create collection
		self._collection = self._client.get_or_create_collection(
			name=self.COLLECTION_NAME,
			metadata={"embed_model": embed_model_name},
		)

	def ensure_manifest(self) -> None:
		"""
		Ensure the collection exists with proper metadata.

		This is called at application startup to verify the index is
		initialized. ChromaDB handles this automatically, so this is
		mainly for API compatibility with the previous implementation.
		"""
		# ChromaDB handles persistence automatically
		# Just verify collection exists and has correct metadata
		if self._collection.metadata.get("embed_model") != self.embed_model_name:
			# Update metadata if embed model changed
			self._collection.modify(metadata={"embed_model": self.embed_model_name})

	def upsert_chunks(
		self,
		file_path: Path,
		md5: str,
		chunks: List[Tuple[int, int, str]],
		embeddings: np.ndarray
	) -> None:
		"""
		Replace all chunks for a given file with new ones.

		This implements an "upsert" (update or insert) operation:
		1. Remove all existing chunks for this file
		2. Add new chunks with fresh embeddings

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
			O(k) where k = number of new chunks
			ChromaDB handles the removal and insertion efficiently
		"""
		# Step 1: Remove old chunks for this file (if any)
		file_path_str = str(file_path)

		# Query existing chunks for this file
		try:
			existing = self._collection.get(
				where={"file_path": file_path_str}
			)
			if existing['ids']:
				self._collection.delete(ids=existing['ids'])
		except Exception:
			# If no existing chunks or error, continue with insert
			pass

		# Step 2: Generate unique IDs for new chunks
		# Use timestamp + index to ensure uniqueness
		timestamp = int(time.time() * 1000000)  # Microseconds
		ids = [f"{timestamp}_{i}" for i in range(len(chunks))]

		# Step 3: Prepare metadata for each chunk
		metadatas = []
		documents = []
		for i, (chunk_start, chunk_end, text) in enumerate(chunks):
			metadatas.append({
				"file_path": file_path_str,
				"chunk_start": chunk_start,
				"chunk_end": chunk_end,
				"md5": md5,
				"added_at": time.time(),
			})
			documents.append(text)

		# Step 4: Insert into ChromaDB
		self._collection.upsert(
			ids=ids,
			embeddings=embeddings.tolist(),
			documents=documents,
			metadatas=metadatas,
		)

	def remove_file(self, file_path: Path) -> None:
		"""
		Remove all chunks associated with a given file from the index.

		This is called when a file is deleted from the vault.

		Args:
			file_path: Path to file whose chunks should be removed

		Example:
			>>> store.remove_file(Path("data/vault/deleted_doc.txt"))
		"""
		file_path_str = str(file_path)

		# Delete all chunks with matching file_path
		try:
			existing = self._collection.get(
				where={"file_path": file_path_str}
			)
			if existing['ids']:
				self._collection.delete(ids=existing['ids'])
		except Exception:
			# File not in index or error - nothing to remove
			pass

	def search(
		self,
		query_embedding: np.ndarray,
		top_k: int = 5
	) -> List[Tuple[IndexEntry, float]]:
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
			Cosine similarity (ChromaDB uses L2 distance by default, but
			for normalized embeddings, L2 and cosine are equivalent)

		Edge Cases:
			- If index is empty, returns []
			- If top_k > index size, returns all chunks

		Example:
			>>> query_emb = embedder.encode_query("What is CurioOS?")
			>>> results = store.search(query_emb, top_k=5)
			>>> for entry, score in results:
			...     print(f"{score:.2f}: {entry.file_path}:{entry.chunk_start}")
		"""
		# Check if collection is empty
		if self._collection.count() == 0:
			return []

		# Clamp top_k to collection size
		k = min(top_k, self._collection.count())

		# Query ChromaDB
		results = self._collection.query(
			query_embeddings=[query_embedding.tolist()],
			n_results=k,
			include=["documents", "metadatas", "distances"]
		)

		# Convert ChromaDB results to IndexEntry format
		output: List[Tuple[IndexEntry, float]] = []

		if not results['ids'] or not results['ids'][0]:
			return []

		for i in range(len(results['ids'][0])):
			doc_id = results['ids'][0][i]
			metadata = results['metadatas'][0][i]
			document = results['documents'][0][i]
			distance = results['distances'][0][i]

			# Convert distance to similarity score
			# ChromaDB uses L2 distance; for normalized vectors:
			# similarity ≈ 1 - (distance^2 / 2)
			# But for better compatibility, we'll use: 1 / (1 + distance)
			similarity = 1.0 / (1.0 + distance)

			# Create IndexEntry (use hash of ID for integer id)
			entry = IndexEntry(
				id=hash(doc_id) % (2**31),  # Convert string ID to int
				file_path=metadata['file_path'],
				chunk_start=int(metadata['chunk_start']),
				chunk_end=int(metadata['chunk_end']),
				md5=metadata['md5'],
				added_at=float(metadata['added_at']),
				text=document,
			)

			output.append((entry, similarity))

		return output

	def get_stats(self) -> dict:
		"""
		Get statistics about the vector store.

		Returns:
			Dictionary with collection statistics
		"""
		return {
			"count": self._collection.count(),
			"embed_model": self.embed_model_name,
			"collection_name": self.COLLECTION_NAME,
		}
