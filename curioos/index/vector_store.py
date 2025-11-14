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
	id: int
	file_path: str
	chunk_start: int
	chunk_end: int
	md5: str
	added_at: float
	text: str


class VectorStore:
	"""Lightweight local vector store persisted to disk (NumPy + JSON)."""

	def __init__(self, index_dir: Path, embed_model_name: str):
		self.index_dir = index_dir
		self.embed_model_name = embed_model_name
		self.embeddings_path = self.index_dir / "embeddings.npy"
		self.index_json_path = self.index_dir / "index.json"
		self.manifest_path = self.index_dir / "manifest.json"

		self._embeddings: Optional[np.ndarray] = None  # shape (N, D)
		self._entries: List[IndexEntry] = []
		self._nn: Optional[NearestNeighbors] = None

		self.index_dir.mkdir(parents=True, exist_ok=True)
		self._load_all()

	def _load_all(self) -> None:
		if self.embeddings_path.exists():
			self._embeddings = np.load(self.embeddings_path)
		else:
			self._embeddings = None

		if self.index_json_path.exists():
			raw = json.loads(self.index_json_path.read_text(encoding="utf-8"))
			self._entries = [IndexEntry(**e) for e in raw]
		else:
			self._entries = []

		self._rebuild_nn()

	def _save_all(self) -> None:
		if self._embeddings is not None:
			np.save(self.embeddings_path, self._embeddings.astype(np.float32, copy=False))
		else:
			if self.embeddings_path.exists():
				self.embeddings_path.unlink(missing_ok=True)

		self.index_json_path.write_text(json.dumps([asdict(e) for e in self._entries], ensure_ascii=False, indent=2), encoding="utf-8")

		manifest = {
			"embed_model": self.embed_model_name,
			"count": len(self._entries),
			"dim": None if self._embeddings is None else int(self._embeddings.shape[1]),
			"updated_at": time.time(),
		}
		self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

	def _rebuild_nn(self) -> None:
		if self._embeddings is None or len(self._entries) == 0:
			self._nn = None
			return
		self._nn = NearestNeighbors(metric="cosine")
		self._nn.fit(self._embeddings)

	def ensure_manifest(self) -> None:
		if not self.manifest_path.exists():
			self._save_all()

	def upsert_chunks(self, file_path: Path, md5: str, chunks: List[Tuple[int, int, str]], embeddings: np.ndarray) -> None:
		"""Replace all chunks for a given file with new ones."""
		# Remove old entries for the file
		old_ids = [e.id for e in self._entries if Path(e.file_path) == file_path]
		if old_ids:
			self._remove_ids(old_ids)

		# Append new entries
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

		if self._embeddings is None or len(self._entries) == 0:
			self._embeddings = embeddings.copy()
			self._entries = new_entries
		else:
			self._embeddings = np.vstack([self._embeddings, embeddings])
			self._entries.extend(new_entries)

		self._save_all()
		self._rebuild_nn()

	def _remove_ids(self, ids: List[int]) -> None:
		if not ids:
			return
		keep_mask = [e.id not in ids for e in self._entries]
		if self._embeddings is not None:
			keep_indices = [i for i, keep in enumerate(keep_mask) if keep]
			if keep_indices:
				self._embeddings = self._embeddings[keep_indices, :]
			else:
				self._embeddings = None
		self._entries = [e for e in self._entries if e.id not in ids]

	def remove_file(self, file_path: Path) -> None:
		rm_ids = [e.id for e in self._entries if Path(e.file_path) == file_path]
		self._remove_ids(rm_ids)
		self._save_all()
		self._rebuild_nn()

	def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[IndexEntry, float]]:
		if self._nn is None or self._embeddings is None or len(self._entries) == 0:
			return []
		k = min(top_k, len(self._entries))
		distances, indices = self._nn.kneighbors([query_embedding], n_neighbors=k, return_distance=True)
		results: List[Tuple[IndexEntry, float]] = []
		for dist, idx in zip(distances[0], indices[0]):
			entry = self._entries[int(idx)]
			# cosine distance -> similarity
			score = 1.0 - float(dist)
			results.append((entry, score))
		return results


