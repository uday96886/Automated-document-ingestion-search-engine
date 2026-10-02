import hashlib
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi


class IngestionEngine:
    """Handles document hashing, parsing, chunking, and indexing into vector store."""

    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        # Initialize fast local embedding model (BAAI/bge-small-en-v1.5)
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Initialize in-memory Qdrant client (can be pointed to server URL)
        self.qdrant = QdrantClient(":memory:")
        
        # Initialize text splitter (500 tokens, 50 overlap)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        
        self._init_collection()

    def _init_collection(self):
        """Creates vector collection if it doesn't exist."""
        self.qdrant.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    def process_and_index(self, doc_id: str, raw_text: str, metadata: Dict[str, Any]):
        """Chunks, embeds, and indexes a single document into Qdrant."""
        chunks = self.text_splitter.split_text(raw_text)
        
        # Generate embeddings for chunks
        embeddings = list(self.embedding_model.embed(chunks))
        
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Deterministic point ID based on doc_id and chunk index
            point_id = hashlib.md5(f"{doc_id}_{idx}".encode()).hexdigest()
            
            chunk_metadata = {
                **metadata,
                "doc_id": doc_id,
                "chunk_index": idx,
                "text": chunk
            }
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=chunk_metadata
                )
            )

        self.qdrant.upsert(collection_name=self.collection_name, points=points)
        print(f" Indexed {len(chunks)} chunks for document '{doc_id}'.")
        return chunks


class HybridSearchEngine:
    """Performs Hybrid Search using Dense Vector Search + Lexical BM25 Search + RRF."""

    def __init__(self, ingestion_engine: IngestionEngine):
        self.ingestion = ingestion_engine
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.bm25 = None

    def build_bm25_index(self, all_chunks_payload: List[Dict[str, Any]]):
        """Builds the BM25 inverted index for lexical search."""
        self.corpus_chunks = all_chunks_payload
        tokenized_corpus = [doc["text"].lower().split() for doc in self.corpus_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("✅ BM25 Lexical Index Built successfully.")

    def dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Vector semantic search via Qdrant."""
        query_vector = list(self.ingestion.embedding_model.embed([query]))[0].tolist()
        
        search_result = self.ingestion.qdrant.search(
            collection_name=self.ingestion.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return [hit.payload for hit in search_result]

    def lexical_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Keyword matching via BM25."""
        if not self.bm25:
            raise ValueError("BM25 index not built yet.")
        
        tokenized_query = query.lower().split()
        top_docs = self.bm25.get_top_n(tokenized_query, self.corpus_chunks, n=top_k)
        return top_docs

    def hybrid_search(self, query: str, top_k: int = 3, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """Combines Dense and Sparse results using Reciprocal Rank Fusion (RRF)."""
        dense_hits = self.dense_search(query, top_k=top_k * 2)
        lexical_hits = self.lexical_search(query, top_k=top_k * 2)

        rrf_scores: Dict[str, float] = {}
        doc_store: Dict[str, Dict[str, Any]] = {}

        # Process dense scores
        for rank, hit in enumerate(dense_hits):
            doc_key = f"{hit['doc_id']}_{hit['chunk_index']}"
            doc_store[doc_key] = hit
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Process lexical scores
        for rank, hit in enumerate(lexical_hits):
            doc_key = f"{hit['doc_id']}_{hit['chunk_index']}"
            doc_store[doc_key] = hit
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + (1.0 / (rrf_k + rank + 1))

        # Sort by hybrid RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        
        results = []
        for key in sorted_keys[:top_k]:
            item = doc_store[key].copy()
            item["rrf_score"] = round(rrf_scores[key], 4)
            results.append(item)
            
        return results


# =====================================================================
#  DEMONSTRATION & RUNTIME EXAMPLE
# =====================================================================
if __name__ == "__main__":
    # 1. Sample Documents
    documents = [
        {
            "id": "doc_001",
            "title": "System Architecture Design",
            "content": """The ingestion pipeline extracts unstructured data using OCR and layout detection models.
                          It splits document text into semantic chunks with sliding overlaps to retain contextual boundaries.
                          Vector embeddings are generated using high-dimensional dense representation transformers."""
        },
        {
            "id": "doc_002",
            "title": "Security & RBAC Policy",
            "content": """Role-Based Access Control (RBAC) ensures users can only search content matching their explicit access privileges.
                          Authentication tokens are validated before querying Qdrant vector databases or Elasticsearch indices.
                          Document encryption is maintained at rest with AES-256 keys."""
        }
    ]

    # 2. Instantiate Engine
    ingestion = IngestionEngine()
    all_payloads = []

    # 3. Process and Index Documents
    for doc in documents:
        chunks = ingestion.process_and_index(
            doc_id=doc["id"],
            raw_text=doc["content"],
            metadata={"title": doc["title"]}
        )
        for idx, chunk in enumerate(chunks):
            all_payloads.append({
                "doc_id": doc["id"],
                "chunk_index": idx,
                "text": chunk,
                "title": doc["title"]
            })

    # 4. Initialize Search & Build Lexical Index
    search_engine = HybridSearchEngine(ingestion)
    search_engine.build_bm25_index(all_payloads)

    # 5. Execute Hybrid Search
    user_query = "How does security and access control work?"
    print(f"\n Query: '{user_query}'\n" + "-"*50)
    
    results = search_engine.hybrid_search(user_query, top_k=2)

    for i, res in enumerate(results, 1):
        print(f"Result #{i} [RRF Score: {res['rrf_score']}]")
        print(f"Document: {res['title']} ({res['doc_id']})")
        print(f"Text Snippet: {res['text'].strip()}")
        print("-" * 50)
