"""
Example implementations of vector store backends for CurioOS
Demonstrates the plugin architecture pattern for supporting multiple vector databases
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from pathlib import Path
import numpy as np
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Unified search result format across all backends"""
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float  # Higher is better (similarity, not distance)


class VectorStoreBackend(ABC):
    """
    Abstract interface for vector storage backends.

    All implementations must support:
    - Adding embeddings with metadata
    - Searching with optional filters
    - Deleting by file path
    - Getting statistics
    - Rebuilding indexes
    """

    @abstractmethod
    def add(self, texts: List[str], embeddings: np.ndarray,
            metadata: List[Dict]) -> None:
        """Add embeddings to the store"""
        pass

    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               filters: Optional[Dict] = None) -> List[SearchResult]:
        """Search for similar embeddings"""
        pass

    @abstractmethod
    def delete_by_file(self, file_path: str) -> int:
        """Delete embeddings for a file. Returns number deleted."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        pass

    @abstractmethod
    def rebuild_index(self) -> None:
        """Rebuild search index (if applicable)"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all data"""
        pass


# ============================================================================
# ChromaDB Backend Implementation
# ============================================================================

class ChromaBackend(VectorStoreBackend):
    """
    ChromaDB backend - Lightweight, embedded vector database

    Best for:
    - Quick prototyping
    - Small to medium datasets (<1M vectors)
    - Easy migration from NumPy

    Installation: pip install chromadb
    """

    def __init__(self, config: Dict[str, Any]):
        import chromadb
        from chromadb.config import Settings

        self.db_path = Path(config.get("CHROMA_DB_PATH", "./data/chroma_db"))
        self.collection_name = config.get("COLLECTION_NAME", "curioos_embeddings")

        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, texts: List[str], embeddings: np.ndarray,
            metadata: List[Dict]) -> None:
        """Add embeddings to ChromaDB"""
        ids = [meta.get('chunk_hash', str(hash(text))) for text, meta in zip(texts, metadata)]

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadata
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               filters: Optional[Dict] = None) -> List[SearchResult]:
        """Search ChromaDB with optional metadata filtering"""

        # ChromaDB uses "where" for filtering
        where_filter = None
        if filters:
            # Convert generic filter format to ChromaDB format
            # Example: {"file_path": "docs/guide.md"} -> {"file_path": "docs/guide.md"}
            where_filter = filters

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter
        )

        # Convert to unified format
        search_results = []
        if results['ids'][0]:  # Check if results exist
            for i in range(len(results['ids'][0])):
                search_results.append(SearchResult(
                    id=results['ids'][0][i],
                    text=results['documents'][0][i],
                    metadata=results['metadatas'][0][i],
                    score=1.0 - results['distances'][0][i]  # Convert distance to similarity
                ))

        return search_results

    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks from a specific file"""
        # ChromaDB delete with filter
        try:
            self.collection.delete(
                where={"file_path": file_path}
            )
            # ChromaDB doesn't return count, so we estimate
            return -1  # Unknown
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            'backend': 'chromadb',
            'vector_count': self.collection.count(),
            'collection_name': self.collection_name,
            'db_path': str(self.db_path)
        }

    def rebuild_index(self) -> None:
        """ChromaDB manages indexes automatically"""
        pass  # No manual rebuild needed

    def clear(self) -> None:
        """Clear all data from collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )


# ============================================================================
# Qdrant Backend Implementation
# ============================================================================

class QdrantBackend(VectorStoreBackend):
    """
    Qdrant backend - Production-grade vector database

    Best for:
    - Production deployments
    - High performance requirements
    - Rich metadata filtering
    - Scalability (millions to billions of vectors)

    Installation:
    - pip install qdrant-client
    - docker run -p 6333:6333 qdrant/qdrant
    """

    def __init__(self, config: Dict[str, Any]):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.url = config.get("QDRANT_URL", "http://localhost:6333")
        self.collection_name = config.get("COLLECTION_NAME", "curioos_embeddings")
        self.vector_size = config.get("VECTOR_SIZE", 384)

        self.client = QdrantClient(url=self.url)

        # Create collection if doesn't exist
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                ),
                # HNSW index configuration for fast ANN search
                hnsw_config={
                    "m": 16,  # Number of edges per node
                    "ef_construct": 100  # Construction time/accuracy tradeoff
                },
                # Optional: Quantization for memory efficiency (97% reduction)
                quantization_config={
                    "scalar": {
                        "type": "int8",
                        "quantile": 0.99,
                        "always_ram": True
                    }
                }
            )

    def add(self, texts: List[str], embeddings: np.ndarray,
            metadata: List[Dict]) -> None:
        """Add embeddings to Qdrant"""
        from qdrant_client.models import PointStruct

        points = []
        for text, embedding, meta in zip(texts, embeddings, metadata):
            point_id = hash(meta.get('chunk_hash', text)) & 0x7FFFFFFFFFFFFFFF

            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    **meta,
                    'text': text
                }
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               filters: Optional[Dict] = None) -> List[SearchResult]:
        """Search Qdrant with optional filtering"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Convert generic filter to Qdrant filter format
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False
        )

        # Convert to unified format
        return [
            SearchResult(
                id=str(hit.id),
                text=hit.payload.get('text', ''),
                metadata={k: v for k, v in hit.payload.items() if k != 'text'},
                score=hit.score
            )
            for hit in results
        ]

    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks from a specific file"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Qdrant filter-based deletion
        delete_filter = Filter(
            must=[
                FieldCondition(
                    key="file_path",
                    match=MatchValue(value=file_path)
                )
            ]
        )

        result = self.client.delete(
            collection_name=self.collection_name,
            points_selector=delete_filter
        )

        # Qdrant doesn't return count in delete, so we estimate
        return -1  # Unknown

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        info = self.client.get_collection(self.collection_name)

        return {
            'backend': 'qdrant',
            'vector_count': info.points_count,
            'indexed_vectors': info.indexed_vectors_count,
            'segments_count': info.segments_count,
            'collection_name': self.collection_name,
            'url': self.url
        }

    def rebuild_index(self) -> None:
        """Qdrant manages indexes automatically"""
        pass  # HNSW index is maintained automatically

    def clear(self) -> None:
        """Clear all data from collection"""
        self.client.delete_collection(self.collection_name)

        # Recreate empty collection
        from qdrant_client.models import Distance, VectorParams
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
        )


# ============================================================================
# LanceDB Backend Implementation
# ============================================================================

class LanceDBBackend(VectorStoreBackend):
    """
    LanceDB backend - Modern columnar vector database

    Best for:
    - Multi-modal data (text, images, etc.)
    - Low memory footprint (disk-based)
    - Modern architecture (Arrow/Lance format)
    - Cloud storage backends (S3)

    Installation: pip install lancedb
    """

    def __init__(self, config: Dict[str, Any]):
        import lancedb

        self.db_path = Path(config.get("LANCEDB_PATH", "./data/lancedb"))
        self.table_name = config.get("TABLE_NAME", "embeddings")

        self.db = lancedb.connect(str(self.db_path))

        # Try to open existing table
        try:
            self.table = self.db.open_table(self.table_name)
        except Exception:
            # Create new table with first data batch
            self.table = None

    def add(self, texts: List[str], embeddings: np.ndarray,
            metadata: List[Dict]) -> None:
        """Add embeddings to LanceDB"""

        # Prepare data in Arrow-compatible format
        data = []
        for text, embedding, meta in zip(texts, embeddings, metadata):
            data.append({
                'id': meta.get('chunk_hash', str(hash(text))),
                'vector': embedding.tolist(),
                'text': text,
                **meta
            })

        if self.table is None:
            # Create table on first add
            self.table = self.db.create_table(
                self.table_name,
                data=data,
                mode="overwrite"
            )
        else:
            # Append to existing table
            self.table.add(data)

        # Create ANN index if table is large enough
        if len(self.table) > 1000:
            try:
                self.table.create_index(
                    metric="cosine",
                    num_partitions=256,
                    num_sub_vectors=96
                )
            except Exception:
                pass  # Index may already exist

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               filters: Optional[Dict] = None) -> List[SearchResult]:
        """Search LanceDB with SQL-like filtering"""

        if self.table is None:
            return []

        query = self.table.search(query_embedding.tolist()) \
            .metric("cosine") \
            .limit(top_k)

        # LanceDB uses SQL-like WHERE clauses
        if filters:
            # Convert dict to SQL WHERE clause
            # Example: {"file_path": "docs/guide.md"} -> "file_path = 'docs/guide.md'"
            conditions = []
            for key, value in filters.items():
                if isinstance(value, str):
                    conditions.append(f"{key} = '{value}'")
                else:
                    conditions.append(f"{key} = {value}")

            where_clause = " AND ".join(conditions)
            query = query.where(where_clause)

        results = query.to_list()

        # Convert to unified format
        return [
            SearchResult(
                id=r['id'],
                text=r['text'],
                metadata={k: v for k, v in r.items()
                         if k not in ['id', 'text', 'vector', '_distance']},
                score=1.0 - r.get('_distance', 0)  # Convert distance to similarity
            )
            for r in results
        ]

    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks from a specific file"""
        if self.table is None:
            return 0

        try:
            # LanceDB uses SQL DELETE
            self.table.delete(f"file_path = '{file_path}'")
            return -1  # Unknown count
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get table statistics"""
        if self.table is None:
            return {
                'backend': 'lancedb',
                'vector_count': 0,
                'table_name': self.table_name,
                'db_path': str(self.db_path)
            }

        return {
            'backend': 'lancedb',
            'vector_count': len(self.table),
            'table_name': self.table_name,
            'db_path': str(self.db_path)
        }

    def rebuild_index(self) -> None:
        """Rebuild ANN index"""
        if self.table and len(self.table) > 0:
            self.table.create_index(
                metric="cosine",
                num_partitions=256,
                num_sub_vectors=96
            )

    def clear(self) -> None:
        """Clear all data from table"""
        if self.table:
            self.db.drop_table(self.table_name)
            self.table = None


# ============================================================================
# Factory Pattern for Backend Selection
# ============================================================================

class VectorStore:
    """
    Factory class for creating vector store backends

    Usage:
        config = {
            "VECTOR_STORE_BACKEND": "qdrant",
            "QDRANT_URL": "http://localhost:6333",
            "COLLECTION_NAME": "my_embeddings"
        }

        store = VectorStore(config)
        store.add(texts, embeddings, metadata)
        results = store.search(query_embedding, top_k=5)
    """

    BACKENDS = {
        'chroma': ChromaBackend,
        'chromadb': ChromaBackend,
        'qdrant': QdrantBackend,
        'lancedb': LanceDBBackend,
        'lance': LanceDBBackend,
    }

    def __init__(self, config: Dict[str, Any]):
        backend_type = config.get("VECTOR_STORE_BACKEND", "chroma").lower()

        if backend_type not in self.BACKENDS:
            raise ValueError(
                f"Unknown backend: {backend_type}. "
                f"Available: {', '.join(self.BACKENDS.keys())}"
            )

        # Instantiate the appropriate backend
        backend_class = self.BACKENDS[backend_type]
        self.backend = backend_class(config)

        print(f"✅ Initialized {backend_type} vector store backend")

    def add(self, texts: List[str], embeddings: np.ndarray,
            metadata: List[Dict]) -> None:
        """Add embeddings to store"""
        return self.backend.add(texts, embeddings, metadata)

    def search(self, query_embedding: np.ndarray, top_k: int = 5,
               filters: Optional[Dict] = None) -> List[SearchResult]:
        """Search for similar embeddings"""
        return self.backend.search(query_embedding, top_k, filters)

    def delete_by_file(self, file_path: str) -> int:
        """Delete embeddings for a file"""
        return self.backend.delete_by_file(file_path)

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return self.backend.get_stats()

    def rebuild_index(self) -> None:
        """Rebuild search index"""
        return self.backend.rebuild_index()

    def clear(self) -> None:
        """Clear all data"""
        return self.backend.clear()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Using ChromaDB backend
    config = {
        "VECTOR_STORE_BACKEND": "chroma",
        "CHROMA_DB_PATH": "./test_chroma_db",
        "COLLECTION_NAME": "test_collection"
    }

    store = VectorStore(config)

    # Add some test data
    texts = ["Hello world", "Python programming", "Machine learning"]
    embeddings = np.random.rand(3, 384).astype(np.float32)
    metadata = [
        {"file_path": "doc1.txt", "chunk_hash": "hash1"},
        {"file_path": "doc2.txt", "chunk_hash": "hash2"},
        {"file_path": "doc1.txt", "chunk_hash": "hash3"},
    ]

    store.add(texts, embeddings, metadata)

    # Search
    query_embedding = np.random.rand(384).astype(np.float32)
    results = store.search(query_embedding, top_k=2)

    print("\nSearch Results:")
    for r in results:
        print(f"  - {r.text[:50]}... (score: {r.score:.3f})")

    # Stats
    print("\nStats:", store.get_stats())

    # Delete by file
    store.delete_by_file("doc1.txt")
    print("\nAfter deletion:", store.get_stats())
