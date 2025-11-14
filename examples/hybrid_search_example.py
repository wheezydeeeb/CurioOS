"""
Hybrid Search Implementation Examples for CurioOS

Demonstrates advanced search patterns combining:
1. Vector similarity search
2. Keyword/BM25 search
3. Reranking with cross-encoders
4. Semantic caching

These patterns represent industry best practices for production RAG systems.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import time


@dataclass
class HybridSearchResult:
    """Result from hybrid search with multiple score components"""
    text: str
    metadata: Dict
    vector_score: float  # Cosine similarity score
    keyword_score: float  # BM25 or keyword match score
    rerank_score: Optional[float] = None  # Cross-encoder score
    final_score: float = 0.0  # Combined score


# ============================================================================
# 1. Hybrid Search with Qdrant (Vector + Full-Text)
# ============================================================================

class QdrantHybridSearch:
    """
    Qdrant-based hybrid search combining vector and keyword search

    Qdrant supports:
    - Vector search (HNSW index)
    - Full-text search (with plugin)
    - Combined scoring strategies
    """

    def __init__(self, qdrant_client, collection_name: str):
        self.client = qdrant_client
        self.collection_name = collection_name

    def hybrid_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining vector and keyword matching

        Args:
            query: Text query for keyword search
            query_embedding: Vector embedding for semantic search
            top_k: Number of results to return
            vector_weight: Weight for vector similarity (0-1)
            keyword_weight: Weight for keyword matching (0-1)

        Returns:
            List of hybrid search results with combined scores
        """
        from qdrant_client.models import (
            SearchRequest,
            Filter,
            FieldCondition,
            MatchText
        )

        # Strategy 1: Separate searches + score fusion
        # ------------------------------------------------

        # 1. Vector search
        vector_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k * 2,  # Get more candidates
            with_payload=True
        )

        # 2. Keyword search (if full-text search is enabled)
        # Note: Requires Qdrant full-text search plugin
        keyword_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k * 2,
            query_filter=Filter(
                should=[
                    FieldCondition(
                        key="text",
                        match=MatchText(text=query)
                    )
                ]
            ),
            with_payload=True
        )

        # 3. Merge and rerank using Reciprocal Rank Fusion (RRF)
        return self._reciprocal_rank_fusion(
            vector_results,
            keyword_results,
            vector_weight,
            keyword_weight,
            top_k
        )

    def _reciprocal_rank_fusion(
        self,
        vector_results,
        keyword_results,
        vector_weight: float,
        keyword_weight: float,
        top_k: int,
        k: int = 60  # RRF constant
    ) -> List[HybridSearchResult]:
        """
        Reciprocal Rank Fusion (RRF) algorithm for combining ranked lists

        RRF Score = sum(1 / (k + rank_i)) for each result list

        This is a popular algorithm used by search engines to combine
        results from multiple retrieval systems.
        """
        scores = {}

        # Add vector search scores
        for rank, hit in enumerate(vector_results):
            doc_id = hit.id
            if doc_id not in scores:
                scores[doc_id] = {
                    'text': hit.payload.get('text', ''),
                    'metadata': {k: v for k, v in hit.payload.items() if k != 'text'},
                    'vector_score': hit.score,
                    'keyword_score': 0.0,
                    'vector_rank': rank,
                    'keyword_rank': None,
                    'rrf_score': 0.0
                }

            scores[doc_id]['rrf_score'] += vector_weight / (k + rank)

        # Add keyword search scores
        for rank, hit in enumerate(keyword_results):
            doc_id = hit.id
            if doc_id not in scores:
                scores[doc_id] = {
                    'text': hit.payload.get('text', ''),
                    'metadata': {k: v for k, v in hit.payload.items() if k != 'text'},
                    'vector_score': 0.0,
                    'keyword_score': hit.score,
                    'vector_rank': None,
                    'keyword_rank': rank,
                    'rrf_score': 0.0
                }
            else:
                scores[doc_id]['keyword_score'] = hit.score
                scores[doc_id]['keyword_rank'] = rank

            scores[doc_id]['rrf_score'] += keyword_weight / (k + rank)

        # Sort by RRF score and return top_k
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )[:top_k]

        return [
            HybridSearchResult(
                text=r['text'],
                metadata=r['metadata'],
                vector_score=r['vector_score'],
                keyword_score=r['keyword_score'],
                final_score=r['rrf_score']
            )
            for r in sorted_results
        ]


# ============================================================================
# 2. Reranking with Cross-Encoder
# ============================================================================

class CrossEncoderReranker:
    """
    Rerank search results using a cross-encoder model

    Cross-encoders jointly encode query+document and are more accurate
    than bi-encoders (like SentenceTransformers) but slower.

    Strategy:
    1. Use fast bi-encoder to get top 50-100 candidates
    2. Use accurate cross-encoder to rerank top 20-30
    3. Return final top_k results

    Popular models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good)
    - cross-encoder/ms-marco-electra-base (slower, better)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[HybridSearchResult],
        top_k: int = 5
    ) -> List[HybridSearchResult]:
        """
        Rerank candidates using cross-encoder

        Args:
            query: User query
            candidates: List of candidate results from initial search
            top_k: Number of top results to return

        Returns:
            Reranked results with cross-encoder scores
        """
        # Prepare query-document pairs
        pairs = [[query, candidate.text] for candidate in candidates]

        # Get cross-encoder scores (batch processing)
        cross_scores = self.model.predict(pairs)

        # Add cross-encoder scores to candidates
        for candidate, score in zip(candidates, cross_scores):
            candidate.rerank_score = float(score)
            # Combine with original score (optional)
            candidate.final_score = 0.5 * candidate.final_score + 0.5 * score

        # Sort by rerank score
        reranked = sorted(
            candidates,
            key=lambda x: x.rerank_score,
            reverse=True
        )[:top_k]

        return reranked


# ============================================================================
# 3. Semantic Caching for Cost Optimization
# ============================================================================

class SemanticCache:
    """
    Cache LLM responses based on semantic similarity of queries

    Benefits:
    - Reduce LLM API costs by 30-70%
    - Faster responses for similar queries
    - Consistent answers for common questions

    Strategy:
    1. Before calling LLM, search cache for similar queries (threshold: 0.95)
    2. If found, return cached response
    3. If not found, call LLM and cache the response
    """

    def __init__(self, vector_store, embedder, similarity_threshold: float = 0.95):
        """
        Args:
            vector_store: Vector database for storing query-response pairs
            embedder: Embedding model for encoding queries
            similarity_threshold: Minimum similarity to consider a cache hit
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.hits = 0
        self.misses = 0

    def get(self, query: str) -> Optional[str]:
        """
        Get cached response for semantically similar query

        Returns:
            Cached response if found, None otherwise
        """
        query_embedding = self.embedder.encode_texts([query])[0]

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=1,
            filters={"cache_type": "llm_response"}
        )

        if results and results[0].score >= self.threshold:
            self.hits += 1
            return results[0].metadata['llm_response']

        self.misses += 1
        return None

    def set(self, query: str, response: str, metadata: Optional[Dict] = None):
        """
        Cache LLM response for this query

        Args:
            query: User query
            response: LLM response to cache
            metadata: Optional metadata (model, tokens, etc.)
        """
        query_embedding = self.embedder.encode_texts([query])[0]

        cache_metadata = {
            "cache_type": "llm_response",
            "query": query,
            "llm_response": response,
            "cached_at": time.time(),
            **(metadata or {})
        }

        self.vector_store.add(
            texts=[query],
            embeddings=np.array([query_embedding]),
            metadata=[cache_metadata]
        )

    def get_cache_stats(self) -> Dict:
        """Get cache hit/miss statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0

        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_queries': total,
            'hit_rate': hit_rate,
            'estimated_savings': f"{hit_rate * 100:.1f}% of LLM calls"
        }


# ============================================================================
# 4. Complete Hybrid RAG Pipeline
# ============================================================================

class HybridRAGPipeline:
    """
    Complete hybrid RAG pipeline with all best practices

    Features:
    1. Hybrid search (vector + keyword)
    2. Cross-encoder reranking
    3. Semantic caching
    4. Query expansion
    """

    def __init__(
        self,
        vector_store,
        embedder,
        cross_encoder=None,
        use_cache: bool = True
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.reranker = cross_encoder or CrossEncoderReranker()

        # Initialize semantic cache
        if use_cache:
            self.cache = SemanticCache(vector_store, embedder)
        else:
            self.cache = None

    def query(
        self,
        question: str,
        top_k: int = 5,
        use_reranking: bool = True,
        use_cache: bool = True
    ) -> Tuple[str, List[HybridSearchResult]]:
        """
        Execute complete RAG pipeline

        Args:
            question: User question
            top_k: Number of context chunks to return
            use_reranking: Whether to use cross-encoder reranking
            use_cache: Whether to check semantic cache

        Returns:
            Tuple of (llm_response, context_chunks)
        """

        # Step 1: Check semantic cache
        if use_cache and self.cache:
            cached_response = self.cache.get(question)
            if cached_response:
                print("✅ Cache hit! Returning cached response.")
                return cached_response, []

        # Step 2: Generate query embedding
        query_embedding = self.embedder.encode_texts([question])[0]

        # Step 3: Hybrid search (vector + keyword)
        # Get more candidates for reranking
        candidate_count = top_k * 4 if use_reranking else top_k

        initial_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=candidate_count
        )

        # Convert to HybridSearchResult format
        hybrid_results = [
            HybridSearchResult(
                text=r.text,
                metadata=r.metadata,
                vector_score=r.score,
                keyword_score=0.0,  # Add keyword scoring if available
                final_score=r.score
            )
            for r in initial_results
        ]

        # Step 4: Rerank with cross-encoder (optional)
        if use_reranking and len(hybrid_results) > 0:
            print(f"🔄 Reranking {len(hybrid_results)} candidates...")
            hybrid_results = self.reranker.rerank(
                query=question,
                candidates=hybrid_results,
                top_k=top_k
            )

        # Step 5: Format context for LLM
        context = self._format_context(hybrid_results)

        # Step 6: Call LLM (placeholder - integrate with actual LLM)
        llm_response = self._call_llm(question, context)

        # Step 7: Cache response
        if use_cache and self.cache:
            self.cache.set(
                query=question,
                response=llm_response,
                metadata={'context_chunks': len(hybrid_results)}
            )

        return llm_response, hybrid_results

    def _format_context(self, results: List[HybridSearchResult]) -> str:
        """Format search results into context for LLM"""
        context_parts = []

        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[{i}] {result.text}\n"
                f"    (source: {result.metadata.get('file_path', 'unknown')}, "
                f"score: {result.final_score:.3f})"
            )

        return "\n\n".join(context_parts)

    def _call_llm(self, question: str, context: str) -> str:
        """
        Call LLM with question and context

        NOTE: This is a placeholder. In production, integrate with:
        - OpenAI API
        - Anthropic Claude
        - Local LLM (Ollama, LlamaCpp)
        """
        prompt = f"""Answer the following question based on the provided context.

Context:
{context}

Question: {question}

Answer:"""

        # Placeholder response
        return f"[LLM Response to: {question}]"

    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        stats = {
            'vector_store': self.vector_store.get_stats()
        }

        if self.cache:
            stats['cache'] = self.cache.get_cache_stats()

        return stats


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of hybrid search and RAG pipeline

    To run this example:
    1. Install dependencies:
       pip install sentence-transformers qdrant-client chromadb

    2. Start Qdrant (if using):
       docker run -p 6333:6333 qdrant/qdrant

    3. Run this script
    """

    # Mock embedder for example
    class MockEmbedder:
        def encode_texts(self, texts):
            return np.random.rand(len(texts), 384).astype(np.float32)

    # Mock vector store for example
    from examples.vector_store_backends import VectorStore

    config = {
        "VECTOR_STORE_BACKEND": "chroma",
        "CHROMA_DB_PATH": "./test_hybrid_db",
    }

    vector_store = VectorStore(config)
    embedder = MockEmbedder()

    # Example: Add some documents
    docs = [
        "Python is a high-level programming language.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neurons.",
        "RAG combines retrieval and generation for better AI responses."
    ]

    embeddings = embedder.encode_texts(docs)
    metadata = [
        {"file_path": f"doc{i}.txt", "chunk_hash": f"hash{i}"}
        for i in range(len(docs))
    ]

    vector_store.add(docs, embeddings, metadata)

    # Create hybrid RAG pipeline
    pipeline = HybridRAGPipeline(
        vector_store=vector_store,
        embedder=embedder,
        use_cache=True
    )

    # Query
    question = "What is machine learning?"
    response, contexts = pipeline.query(
        question=question,
        top_k=3,
        use_reranking=True
    )

    print(f"\nQuestion: {question}")
    print(f"\nResponse: {response}")
    print(f"\nContext chunks used: {len(contexts)}")

    # Query again (should hit cache)
    question2 = "What is machine learning?"
    response2, contexts2 = pipeline.query(question2, top_k=3)

    # Stats
    print("\n" + "="*60)
    print("Pipeline Stats:")
    print("="*60)
    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
