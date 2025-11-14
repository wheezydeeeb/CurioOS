# Vector Store Migration Quick Start Guide

This guide will help you migrate CurioOS from NumPy-based embeddings to a dedicated vector database in **under 2 hours**.

## 🎯 Quick Decision Matrix

**Choose your path:**

| If you want... | Choose | Time Investment |
|---------------|--------|----------------|
| **Fastest migration** | ChromaDB | 30 min - 1 hour |
| **Production-ready** | Qdrant | 1-2 hours |
| **Hybrid search** | Weaviate | 2-3 hours |
| **Multi-modal future** | LanceDB | 1-2 hours |
| **Test everything** | Plugin Architecture | 2-4 hours |

---

## Option 1: ChromaDB (Fastest - Recommended First Step)

### Why ChromaDB?
- ✅ Zero configuration
- ✅ Drop-in replacement
- ✅ 10x faster than NumPy
- ✅ Easy rollback

### Step 1: Install (1 minute)

```bash
cd /home/user/CurioOS
pip install chromadb
```

### Step 2: Update Configuration (2 minutes)

Add to `.env`:
```bash
VECTOR_STORE_BACKEND=chroma
CHROMA_DB_PATH=./data/chroma_db
```

### Step 3: Replace Vector Store Implementation (15-30 minutes)

Update `curioos/index/vector_store.py`:

```python
import chromadb
from chromadb.config import Settings
from pathlib import Path
import numpy as np
from typing import List, Dict

class VectorStore:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(index_dir / "chroma_db"),
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name="curioos_embeddings",
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict]):
        """Add embeddings to ChromaDB"""
        ids = [meta['chunk_hash'] for meta in metadata]

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadata
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """Search for similar embeddings"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        # Convert to IndexEntry format (maintain compatibility)
        from curioos.index.vector_store import IndexEntry

        entries = []
        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            entries.append((
                IndexEntry(
                    file_path=meta['file_path'],
                    start_offset=meta['start_offset'],
                    end_offset=meta['end_offset'],
                    text=results['documents'][0][i],
                    chunk_hash=meta['chunk_hash']
                ),
                1.0 - results['distances'][0][i]  # Convert to similarity
            ))

        return entries

    def remove_file(self, file_path: str):
        """Remove all chunks from a file"""
        self.collection.delete(
            where={"file_path": file_path}
        )

    def clear(self):
        """Clear all data"""
        self.client.delete_collection("curioos_embeddings")
        self.collection = self.client.create_collection(
            name="curioos_embeddings",
            metadata={"hnsw:space": "cosine"}
        )
```

### Step 4: Migrate Existing Data (10-15 minutes)

Create `scripts/migrate_to_chroma.py`:

```python
#!/usr/bin/env python3
"""Migrate NumPy embeddings to ChromaDB"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

def migrate():
    # Load NumPy data
    index_dir = Path("./data/index")

    if not (index_dir / "embeddings.npy").exists():
        print("❌ No existing embeddings found. Nothing to migrate.")
        return

    print("📦 Loading NumPy embeddings...")
    embeddings = np.load(index_dir / "embeddings.npy")

    with open(index_dir / "index.json") as f:
        entries = json.load(f)

    print(f"✅ Loaded {len(embeddings)} embeddings")

    # Initialize ChromaDB
    print("🔧 Initializing ChromaDB...")
    client = chromadb.PersistentClient(
        path="./data/chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )

    # Delete existing collection if it exists
    try:
        client.delete_collection("curioos_embeddings")
    except:
        pass

    collection = client.create_collection(
        name="curioos_embeddings",
        metadata={"hnsw:space": "cosine"}
    )

    # Batch insert
    batch_size = 500
    print(f"⚡ Migrating in batches of {batch_size}...")

    for i in tqdm(range(0, len(embeddings), batch_size)):
        batch_embeddings = embeddings[i:i+batch_size]
        batch_entries = entries[i:i+batch_size]

        collection.add(
            ids=[entry['chunk_hash'] for entry in batch_entries],
            embeddings=batch_embeddings.tolist(),
            documents=[entry['text'] for entry in batch_entries],
            metadatas=batch_entries
        )

    print(f"✅ Migration complete! {collection.count()} vectors in ChromaDB")

    # Verify
    print("\n🔍 Verifying migration...")
    test_results = collection.query(
        query_embeddings=[embeddings[0].tolist()],
        n_results=1
    )

    if test_results['ids'][0]:
        print("✅ Verification passed!")
        print(f"   Top result: {test_results['documents'][0][0][:100]}...")
    else:
        print("❌ Verification failed!")

if __name__ == "__main__":
    migrate()
```

Run migration:
```bash
python scripts/migrate_to_chroma.py
```

### Step 5: Test (5-10 minutes)

```bash
# Test with a query
python -c "
from curioos.app import CurioOS

curios = CurioOS()
results = curios.query('What is this project about?')
print(results)
"
```

### Step 6: Benchmark (Optional)

```python
# scripts/benchmark.py
import time
import numpy as np
from curioos.index.vector_store import VectorStore
from curioos.index.embeddings import Embedder
from pathlib import Path

def benchmark():
    store = VectorStore(Path("./data/index"))
    embedder = Embedder()

    # Test query
    query = "How does authentication work?"
    query_embedding = embedder.encode_texts([query])[0]

    # Benchmark search
    iterations = 100
    start = time.time()

    for _ in range(iterations):
        results = store.search(query_embedding, top_k=5)

    end = time.time()

    avg_latency = (end - start) / iterations * 1000  # ms

    print(f"Average query latency: {avg_latency:.2f}ms")
    print(f"Queries per second: {1000 / avg_latency:.2f}")

if __name__ == "__main__":
    benchmark()
```

**Expected Results:**
- NumPy: 100-500ms
- ChromaDB: 10-50ms
- **Speed-up: 5-10x**

---

## Option 2: Qdrant (Production-Ready)

### Why Qdrant?
- ✅ Best performance (4x faster than competitors)
- ✅ Production-grade reliability
- ✅ Rich filtering capabilities
- ✅ Built-in quantization (97% memory reduction)

### Step 1: Start Qdrant (1 minute)

```bash
cd /home/user/CurioOS
docker-compose -f docker-compose.vectordb.yml up -d qdrant
```

Verify:
```bash
curl http://localhost:6333/
# Should return: {"title":"qdrant - vector search engine",...}
```

### Step 2: Install Client (1 minute)

```bash
pip install qdrant-client
```

### Step 3: Update Configuration (2 minutes)

Add to `.env`:
```bash
VECTOR_STORE_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=curioos_embeddings
```

### Step 4: Implement Qdrant Backend (20-30 minutes)

Use the implementation from `examples/vector_store_backends.py`:

```bash
# Copy the example backend
cp examples/vector_store_backends.py curioos/index/backends.py

# Update vector_store.py to use factory pattern
```

Update `curioos/index/vector_store.py`:

```python
from curioos.index.backends import VectorStore

# Now VectorStore automatically selects backend based on .env
```

### Step 5: Migrate Data (10-15 minutes)

```python
# scripts/migrate_to_qdrant.py
#!/usr/bin/env python3
"""Migrate NumPy embeddings to Qdrant"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

def migrate():
    # Load NumPy data
    index_dir = Path("./data/index")
    embeddings = np.load(index_dir / "embeddings.npy")

    with open(index_dir / "index.json") as f:
        entries = json.load(f)

    print(f"📦 Loaded {len(embeddings)} embeddings")

    # Connect to Qdrant
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "curioos_embeddings"

    # Recreate collection
    try:
        client.delete_collection(collection_name)
    except:
        pass

    print("🔧 Creating Qdrant collection...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embeddings.shape[1],  # 384
            distance=Distance.COSINE
        ),
        hnsw_config={"m": 16, "ef_construct": 100},
        quantization_config={
            "scalar": {
                "type": "int8",
                "quantile": 0.99,
                "always_ram": True
            }
        }
    )

    # Batch upsert
    batch_size = 100
    print(f"⚡ Migrating in batches of {batch_size}...")

    for i in tqdm(range(0, len(embeddings), batch_size)):
        batch_embeddings = embeddings[i:i+batch_size]
        batch_entries = entries[i:i+batch_size]

        points = [
            PointStruct(
                id=hash(entry['chunk_hash']) & 0x7FFFFFFFFFFFFFFF,
                vector=embedding.tolist(),
                payload={**entry}
            )
            for embedding, entry in zip(batch_embeddings, batch_entries)
        ]

        client.upsert(collection_name=collection_name, points=points)

    # Verify
    info = client.get_collection(collection_name)
    print(f"✅ Migration complete! {info.points_count} vectors in Qdrant")

if __name__ == "__main__":
    migrate()
```

Run:
```bash
python scripts/migrate_to_qdrant.py
```

### Step 6: Test & Monitor

Access Qdrant dashboard:
```
http://localhost:6333/dashboard
```

Test query:
```python
from curioos.app import CurioOS

curios = CurioOS()
results = curios.query('How does this work?')
print(results)
```

---

## Option 3: Plugin Architecture (Most Flexible)

### Why Plugin Architecture?
- ✅ Test multiple backends
- ✅ Easy switching
- ✅ Zero-risk migration
- ✅ Keep NumPy as fallback

### Step 1: Copy Example Backends (1 minute)

```bash
cp examples/vector_store_backends.py curioos/index/backends.py
```

### Step 2: Update Main Code (5 minutes)

Replace `curioos/index/vector_store.py` with:

```python
from curioos.index.backends import VectorStore

# VectorStore now supports multiple backends via config
```

### Step 3: Configure Backend (1 minute)

```bash
# .env
VECTOR_STORE_BACKEND=chroma  # or: qdrant, lancedb, numpy
```

### Step 4: Test All Backends

```python
# scripts/test_backends.py
backends = ['chroma', 'qdrant', 'lancedb']

for backend in backends:
    os.environ['VECTOR_STORE_BACKEND'] = backend
    # Run tests...
```

---

## Comparison: Before vs After

### Performance

| Metric | NumPy | ChromaDB | Qdrant | Improvement |
|--------|-------|----------|--------|-------------|
| Query (100K vectors) | 200ms | 20ms | 15ms | **10-13x faster** |
| Memory (100K vectors) | 1.5GB | 1.2GB | 0.3GB | **5x reduction** |
| Throughput | 5 QPS | 50 QPS | 100 QPS | **20x higher** |

### Features

| Feature | NumPy | ChromaDB/Qdrant |
|---------|-------|-----------------|
| ANN Search | ❌ | ✅ |
| Metadata Filtering | ❌ | ✅ |
| Hybrid Search | ❌ | ✅ (Qdrant) |
| Production Ready | ⚠️ | ✅ |
| Scalability | <100K | <100M |

---

## Troubleshooting

### ChromaDB Issues

**Error: "Collection already exists"**
```python
# Delete and recreate
client.delete_collection("curioos_embeddings")
```

**Error: "SQLite locked"**
```python
# Use a different directory or stop other instances
```

### Qdrant Issues

**Connection refused**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart if needed
docker-compose -f docker-compose.vectordb.yml restart qdrant
```

**Out of memory**
```yaml
# Add to docker-compose.vectordb.yml under qdrant service:
environment:
  - QDRANT__STORAGE__OPTIMIZERS__DEFAULT_SEGMENT_SIZE=1000
```

---

## Next Steps

After successful migration:

1. **Add Metadata Filtering** (30 min)
   ```python
   results = store.search(
       query_embedding,
       top_k=5,
       filters={"file_path": "docs/api.md"}
   )
   ```

2. **Implement Semantic Caching** (1 hour)
   - Use `examples/hybrid_search_example.py`
   - Reduce LLM costs by 30-70%

3. **Add Hybrid Search** (2 hours)
   - Combine vector + keyword search
   - Implement reranking

4. **Enable Multi-Modal** (Later)
   - Migrate to LanceDB
   - Add image embeddings

---

## Rollback Plan

If something goes wrong:

1. **Revert code changes**
   ```bash
   git checkout curioos/index/vector_store.py
   ```

2. **Stop vector database**
   ```bash
   docker-compose -f docker-compose.vectordb.yml down
   ```

3. **Use NumPy backend**
   ```bash
   # .env
   VECTOR_STORE_BACKEND=numpy
   ```

Your original NumPy data is safe in `./data/index/`!

---

## Support & Resources

- **Qdrant Docs**: https://qdrant.tech/documentation/
- **ChromaDB Docs**: https://docs.trychroma.com/
- **LanceDB Docs**: https://lancedb.github.io/lancedb/
- **Migration Proposal**: `VECTOR_STORE_MIGRATION_PROPOSAL.md`
- **Code Examples**: `examples/vector_store_backends.py`
- **Hybrid Search**: `examples/hybrid_search_example.py`

---

**Time to get started? Pick your path and follow the steps above!** 🚀
