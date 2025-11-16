#!/usr/bin/env python3
"""
Test streaming functionality.
"""

import sys
from pathlib import Path

print("="*60)
print("Testing Streaming Functionality")
print("="*60)

# Import required modules
from curioos.config import load_config
from curioos.index.embeddings import Embedder
from curioos.index.chroma_store import ChromaVectorStore
from curioos.llm.groq_client import GroqClient
from curioos.rag.graph import build_graph

print("\n[1] Initializing components...")
cfg = load_config()
embedder = Embedder(cfg.embed_model, cache_folder=cfg.models_cache_dir)
store = ChromaVectorStore(cfg.index_dir, embedder.model_name)
groq = GroqClient(cfg.groq_api_key, cfg.groq_model)
checkpoint_path = str(cfg.index_dir / "checkpoints.db")
graph = build_graph(store, embedder, groq, checkpoint_path=checkpoint_path)
print("✅ Components initialized")

print("\n[2] Testing streaming with 'values' mode...")
try:
    state = {"question": "test streaming"}
    config = {"configurable": {"thread_id": "test_stream"}}

    chunk_count = 0
    received_contexts = False
    received_answer = False

    for chunk in graph.stream(state, config=config, stream_mode="values"):
        chunk_count += 1
        print(f"   Chunk {chunk_count}: {list(chunk.keys())}")

        if "contexts" in chunk and chunk.get("contexts"):
            received_contexts = True
        if "answer" in chunk and chunk.get("answer"):
            received_answer = True

    print(f"✅ Streaming completed")
    print(f"   - Total chunks: {chunk_count}")
    print(f"   - Received contexts: {received_contexts}")
    print(f"   - Received answer: {received_answer}")

    if chunk_count > 1 and received_contexts and received_answer:
        print("✅ STREAMING WORKING!")
    else:
        print("⚠️  Streaming may have issues")

except Exception as e:
    print(f"❌ Streaming test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n[3] Testing streaming with 'updates' mode...")
try:
    state = {"question": "test updates"}
    config = {"configurable": {"thread_id": "test_updates"}}

    update_count = 0
    for update in graph.stream(state, config=config, stream_mode="updates"):
        update_count += 1
        # Updates show which node produced each update
        for node_name, node_update in update.items():
            if node_update is not None:
                print(f"   Update {update_count} from '{node_name}': {list(node_update.keys())}")
            else:
                print(f"   Update {update_count} from '{node_name}': (empty)")

    print(f"✅ Updates mode completed")
    print(f"   - Total updates: {update_count}")

except Exception as e:
    print(f"❌ Updates mode test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Streaming Test Complete")
print("="*60)
