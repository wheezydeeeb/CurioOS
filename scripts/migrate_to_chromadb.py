#!/usr/bin/env python3
"""
Migration Script: NumPy+JSON Index to ChromaDB

This script migrates the existing CurioOS vector index from the NumPy+JSON
format to ChromaDB format.

Usage:
	python scripts/migrate_to_chromadb.py

The script will:
1. Read existing index.json and embeddings.npy
2. Create a new ChromaDB collection
3. Populate it with all chunks and embeddings
4. Backup old files to data/index_backup/
5. Update the index to use ChromaDB

Requirements:
	- Must be run from CurioOS root directory
	- Existing index must be in data/index/
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from curioos.index.chroma_store import ChromaVectorStore


def migrate_to_chromadb(
	old_index_dir: Path,
	new_index_dir: Path,
	backup_dir: Path
) -> None:
	"""
	Migrate from NumPy+JSON format to ChromaDB.

	Args:
		old_index_dir: Directory containing old index files
		new_index_dir: Directory for new ChromaDB storage
		backup_dir: Directory to backup old files
	"""
	print("=" * 60)
	print("CurioOS Index Migration: NumPy+JSON → ChromaDB")
	print("=" * 60)

	# Step 1: Verify old index exists
	embeddings_path = old_index_dir / "embeddings.npy"
	index_json_path = old_index_dir / "index.json"
	manifest_path = old_index_dir / "manifest.json"

	if not index_json_path.exists():
		print("❌ Error: index.json not found in", old_index_dir)
		print("   Nothing to migrate. Is the index directory correct?")
		return

	print(f"\n📂 Reading old index from: {old_index_dir}")

	# Step 2: Load manifest to get embed model name
	embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Default
	if manifest_path.exists():
		manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
		embed_model_name = manifest.get("embed_model", embed_model_name)
		print(f"   Embed model: {embed_model_name}")
		print(f"   Chunk count: {manifest.get('count', 'unknown')}")
		print(f"   Dimensions: {manifest.get('dim', 'unknown')}")

	# Step 3: Load index entries
	print("\n📖 Loading index entries...")
	index_data = json.loads(index_json_path.read_text(encoding="utf-8"))
	print(f"   Found {len(index_data)} chunks")

	# Step 4: Load embeddings
	embeddings = None
	if embeddings_path.exists():
		print("\n🔢 Loading embeddings...")
		embeddings = np.load(embeddings_path)
		print(f"   Shape: {embeddings.shape}")

		# Verify dimensions match
		if len(index_data) != embeddings.shape[0]:
			print(f"   ⚠️  Warning: Mismatch between index entries ({len(index_data)}) "
			      f"and embeddings ({embeddings.shape[0]})")
	else:
		print("\n⚠️  No embeddings.npy found - will create empty ChromaDB")

	# Step 5: Create new ChromaDB store
	print(f"\n🔨 Creating ChromaDB store in: {new_index_dir}")
	store = ChromaVectorStore(new_index_dir, embed_model_name)

	# Step 6: Migrate data file by file
	if index_data and embeddings is not None:
		print("\n📦 Migrating chunks to ChromaDB...")

		# Group chunks by file for efficient upsert
		file_chunks = {}
		for i, entry in enumerate(index_data):
			file_path = entry['file_path']
			if file_path not in file_chunks:
				file_chunks[file_path] = {
					'chunks': [],
					'embeddings': [],
					'md5': entry['md5']
				}

			file_chunks[file_path]['chunks'].append((
				entry['chunk_start'],
				entry['chunk_end'],
				entry['text']
			))
			file_chunks[file_path]['embeddings'].append(embeddings[i])

		# Upsert each file
		for file_path, data in file_chunks.items():
			chunks = data['chunks']
			chunk_embeddings = np.array(data['embeddings'])
			md5 = data['md5']

			print(f"   - {Path(file_path).name}: {len(chunks)} chunks")
			store.upsert_chunks(Path(file_path), md5, chunks, chunk_embeddings)

		print(f"\n✅ Successfully migrated {len(index_data)} chunks")

		# Verify migration
		stats = store.get_stats()
		print(f"   ChromaDB stats: {stats['count']} chunks in collection")

	# Step 7: Backup old files
	if backup_dir:
		print(f"\n💾 Backing up old index to: {backup_dir}")
		backup_dir.mkdir(parents=True, exist_ok=True)

		for file_path in [embeddings_path, index_json_path, manifest_path]:
			if file_path.exists():
				backup_path = backup_dir / file_path.name
				shutil.copy2(file_path, backup_path)
				print(f"   ✓ Backed up {file_path.name}")

	print("\n" + "=" * 60)
	print("✨ Migration complete!")
	print("=" * 60)
	print("\nNext steps:")
	print("1. Test the new ChromaDB index with your application")
	print("2. If everything works, you can delete the backup")
	print(f"   Backup location: {backup_dir}")
	print("\nTo rollback:")
	print(f"1. Delete {new_index_dir}")
	print(f"2. Restore files from {backup_dir} to {old_index_dir}")


def main():
	"""Run the migration."""
	# Define paths
	project_root = Path(__file__).parent.parent
	old_index_dir = project_root / "data" / "index"
	new_index_dir = project_root / "data" / "chroma"
	backup_dir = project_root / "data" / "index_backup"

	# Check if migration already done
	if new_index_dir.exists():
		print(f"⚠️  ChromaDB directory already exists: {new_index_dir}")
		response = input("   Delete and re-migrate? (y/N): ").strip().lower()
		if response == 'y':
			shutil.rmtree(new_index_dir)
			print("   Deleted existing ChromaDB directory")
		else:
			print("   Aborted. Exiting...")
			return

	# Run migration
	try:
		migrate_to_chromadb(old_index_dir, new_index_dir, backup_dir)
	except Exception as e:
		print(f"\n❌ Migration failed: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)


if __name__ == "__main__":
	main()
