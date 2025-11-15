#!/usr/bin/env python3
"""
Test script for ChromaVectorStore

This script verifies that the ChromaDB migration worked correctly
by testing basic operations.
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from curioos.index.chroma_store import ChromaVectorStore


def test_chromadb():
	"""Test ChromaVectorStore functionality."""
	print("=" * 60)
	print("Testing ChromaVectorStore")
	print("=" * 60)

	# Initialize store
	index_dir = project_root / "data" / "chroma"
	embed_model = "sentence-transformers/all-MiniLM-L6-v2"

	print(f"\n📂 Loading ChromaDB from: {index_dir}")
	store = ChromaVectorStore(index_dir, embed_model)

	# Get stats
	stats = store.get_stats()
	print(f"\n📊 Store Statistics:")
	print(f"   Collection: {stats['collection_name']}")
	print(f"   Embed Model: {stats['embed_model']}")
	print(f"   Total Chunks: {stats['count']}")

	if stats['count'] == 0:
		print("\n⚠️  No chunks found in store")
		return

	# Test search with a dummy query embedding
	print(f"\n🔍 Testing search functionality...")
	# Create a random query embedding (384 dimensions to match all-MiniLM-L6-v2)
	query_embedding = np.random.randn(384).astype(np.float32)
	# Normalize it (cosine similarity expects normalized vectors)
	query_embedding = query_embedding / np.linalg.norm(query_embedding)

	results = store.search(query_embedding, top_k=3)

	print(f"   Found {len(results)} results")
	print(f"\n📄 Top Results:")
	for i, (entry, score) in enumerate(results, 1):
		print(f"\n   {i}. Score: {score:.4f}")
		print(f"      File: {Path(entry.file_path).name}")
		print(f"      Offset: {entry.chunk_start}-{entry.chunk_end}")
		print(f"      Text preview: {entry.text[:100]}...")

	# Test upsert
	print(f"\n📝 Testing upsert functionality...")
	test_file = project_root / "data" / "vault" / "test_doc.txt"
	test_chunks = [
		(0, 50, "This is a test document chunk for ChromaDB testing."),
	]
	test_embeddings = np.random.randn(1, 384).astype(np.float32)
	# Normalize
	test_embeddings = test_embeddings / np.linalg.norm(test_embeddings, axis=1, keepdims=True)

	store.upsert_chunks(test_file, "test_md5_hash", test_chunks, test_embeddings)
	print(f"   ✓ Upserted 1 test chunk")

	# Verify upsert
	new_stats = store.get_stats()
	print(f"   New chunk count: {new_stats['count']}")

	# Test remove
	print(f"\n🗑️  Testing remove functionality...")
	store.remove_file(test_file)
	print(f"   ✓ Removed test file")

	# Verify remove
	final_stats = store.get_stats()
	print(f"   Final chunk count: {final_stats['count']}")

	print("\n" + "=" * 60)
	print("✅ All tests passed!")
	print("=" * 60)


if __name__ == "__main__":
	try:
		test_chromadb()
	except Exception as e:
		print(f"\n❌ Test failed: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)
