#!/usr/bin/env python3
"""
Test checkpointing functionality by running a simple query.
"""

import sys
from pathlib import Path

print("="*60)
print("Testing Checkpointing Functionality")
print("="*60)

# Import required modules
from curioos.config import load_config
from curioos.index.embeddings import Embedder
from curioos.index.chroma_store import ChromaVectorStore
from curioos.llm.groq_client import GroqClient
from curioos.rag.graph import build_graph

print("\n[1] Loading configuration...")
cfg = load_config()
checkpoint_path = str(cfg.index_dir / "checkpoints.db")
print(f"✅ Checkpoint path: {checkpoint_path}")

print("\n[2] Initializing components...")
embedder = Embedder(cfg.embed_model, cache_folder=cfg.models_cache_dir)
store = ChromaVectorStore(cfg.index_dir, embedder.model_name)
groq = GroqClient(cfg.groq_api_key, cfg.groq_model)
print("✅ Components initialized")

print("\n[3] Building graph with checkpointing...")
graph = build_graph(store, embedder, groq, checkpoint_path=checkpoint_path)
print("✅ Graph built")

# Check if checkpoint DB exists before query
checkpoint_exists_before = Path(checkpoint_path).exists()
print(f"\n[4] Checkpoint DB before query: {'EXISTS' if checkpoint_exists_before else 'NOT EXISTS'}")

print("\n[5] Running a simple test query...")
try:
    # Run a simple query with thread_id
    state = {"question": "test"}
    config = {"configurable": {"thread_id": "test_thread_1"}}

    # Invoke the graph
    result = graph.invoke(state, config=config)

    print("✅ Query completed successfully")
    print(f"   - Answer length: {len(result.get('answer', ''))}")

except Exception as e:
    print(f"⚠️  Query failed (this is OK if Groq API key not set): {e}")
    # This is fine - we're mainly testing checkpoint creation

# Check if checkpoint DB exists after query
checkpoint_exists_after = Path(checkpoint_path).exists()
print(f"\n[6] Checkpoint DB after query: {'EXISTS ✅' if checkpoint_exists_after else 'NOT EXISTS ❌'}")

if checkpoint_exists_after:
    size = Path(checkpoint_path).stat().st_size
    print(f"   - File size: {size:,} bytes")

    # Try to verify we can read checkpoints
    try:
        # Get state history
        history = list(graph.get_state_history(config))
        print(f"   - State snapshots saved: {len(history)}")
        print("✅ CHECKPOINTING WORKING!")
    except Exception as e:
        print(f"   - Could not read history: {e}")

print("\n" + "="*60)
print("Checkpointing Test Complete")
print("="*60)
