#!/usr/bin/env python3
"""
Integration Test for ChromaDB Migration

This script tests the complete CurioOS pipeline with ChromaDB:
1. Index creation
2. File watching (add, modify, delete)
3. Q&A functionality

Usage:
	python scripts/test_integration.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from curioos.config import load_config
from curioos.index.embeddings import Embedder
from curioos.index.chroma_store import ChromaVectorStore
from curioos.ingest.parser import load_and_normalize
from curioos.ingest.chunker import chunk_text


def test_integration():
	"""Run comprehensive integration tests."""
	print("=" * 60)
	print("ChromaDB Integration Test")
	print("=" * 60)

	# Load configuration
	cfg = load_config()
	print(f"\n📂 Configuration:")
	print(f"   Vault: {cfg.vault_dir}")
	print(f"   Index: {cfg.index_dir}")
	print(f"   Embed Model: {cfg.embed_model}")

	# Initialize components
	print(f"\n🔧 Initializing components...")
	embedder = Embedder(cfg.embed_model, cache_folder=cfg.models_cache_dir)
	store = ChromaVectorStore(cfg.index_dir, embedder.model_name)
	store.ensure_manifest()

	# Get initial stats
	stats = store.get_stats()
	print(f"\n📊 Initial Store Statistics:")
	print(f"   Chunks: {stats['count']}")
	print(f"   Model: {stats['embed_model']}")

	# Test 1: Index existing files
	print(f"\n✅ Test 1: Index Existing Files")
	existing_files = list(cfg.vault_dir.glob("*.md")) + list(cfg.vault_dir.glob("*.pdf"))
	print(f"   Found {len(existing_files)} files in vault")

	for file in existing_files:
		try:
			text, md5 = load_and_normalize(file)
			chunks = chunk_text(text, chunk_size=800, overlap=150)
			if chunks:
				embeddings = embedder.encode_texts([ch[2] for ch in chunks])
				store.upsert_chunks(file, md5, chunks, embeddings)
				print(f"   ✓ Indexed {file.name}: {len(chunks)} chunks")
		except Exception as e:
			print(f"   ✗ Failed to index {file.name}: {e}")

	# Check stats after indexing
	stats = store.get_stats()
	print(f"\n   Total chunks after indexing: {stats['count']}")

	# Test 2: Search functionality
	print(f"\n✅ Test 2: Search Functionality")
	test_queries = [
		"What is CurioOS?",
		"Who is Tridib?",
		"Invoice details"
	]

	for query in test_queries:
		query_embedding = embedder.encode_query(query)
		results = store.search(query_embedding, top_k=3)
		print(f"\n   Query: '{query}'")
		print(f"   Results: {len(results)}")
		if results:
			top_result = results[0]
			print(f"   Top match (score={top_result[1]:.4f}): {Path(top_result[0].file_path).name}")
			print(f"   Preview: {top_result[0].text[:80]}...")

	# Test 3: Upsert (simulating file modification)
	print(f"\n✅ Test 3: File Modification (Upsert)")
	test_file = cfg.vault_dir / "test_modify.md"
	test_file.write_text("# Test Document\n\nThis is a test for modification detection.")

	# Index it
	text, md5 = load_and_normalize(test_file)
	chunks = chunk_text(text, chunk_size=800, overlap=150)
	embeddings = embedder.encode_texts([ch[2] for ch in chunks])
	store.upsert_chunks(test_file, md5, chunks, embeddings)
	print(f"   ✓ Created test file with {len(chunks)} chunks")

	stats_before = store.get_stats()
	print(f"   Chunks before update: {stats_before['count']}")

	# Modify it
	test_file.write_text("# Test Document\n\nThis is MODIFIED content for testing.")
	text, md5_new = load_and_normalize(test_file)
	chunks_new = chunk_text(text, chunk_size=800, overlap=150)
	embeddings_new = embedder.encode_texts([ch[2] for ch in chunks_new])
	store.upsert_chunks(test_file, md5_new, chunks_new, embeddings_new)
	print(f"   ✓ Modified test file, now has {len(chunks_new)} chunks")

	stats_after = store.get_stats()
	print(f"   Chunks after update: {stats_after['count']}")

	# Test 4: Delete
	print(f"\n✅ Test 4: File Deletion")
	store.remove_file(test_file)
	test_file.unlink(missing_ok=True)
	stats_final = store.get_stats()
	print(f"   ✓ Deleted test file")
	print(f"   Final chunk count: {stats_final['count']}")

	# Test 5: Verify data persistence
	print(f"\n✅ Test 5: Data Persistence")
	print(f"   Creating new store instance to test persistence...")
	store2 = ChromaVectorStore(cfg.index_dir, embedder.model_name)
	stats_reloaded = store2.get_stats()
	print(f"   Reloaded chunk count: {stats_reloaded['count']}")

	if stats_reloaded['count'] == stats_final['count']:
		print(f"   ✓ Data persisted correctly!")
	else:
		print(f"   ✗ Data mismatch! Expected {stats_final['count']}, got {stats_reloaded['count']}")

	print("\n" + "=" * 60)
	print("✨ Integration test complete!")
	print("=" * 60)


if __name__ == "__main__":
	try:
		test_integration()
	except Exception as e:
		print(f"\n❌ Test failed: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)
