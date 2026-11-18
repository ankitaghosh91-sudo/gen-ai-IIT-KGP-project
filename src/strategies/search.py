
"""Search strategies (keyword and Chroma)."""

import os
import re

from src.models.records import Document, Evidence ,Chunk
from src.strategies.base import ChunkingStrategyBase, SearchStrategyBase
from src.strategies.chunking import WholeDocumentChunking
from src.strategies.embeddings import HashingEmbedder
from rank_bm25 import BM25Okapi
import logging
from pathlib import Path

# Dedicated ChromaDB diagnostic logger
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("chroma_retrieval")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    file_handler = logging.FileHandler(
        LOG_DIR / "chroma_retrieval.log",
        encoding="utf-8",
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def build_chunks(
    documents: list[Document],
    chunker: ChunkingStrategyBase | None = None,
) -> list[Chunk]:
    """Split documents into chunk records, supporting parent-child hierarchy if available."""

    use_chunker = chunker or WholeDocumentChunking()
    chunks = []

    # Check if the chunker supports parent-child hierarchy
    if hasattr(use_chunker, "split_with_hierarchy"):
        for document in documents:
            hierarchy = use_chunker.split_with_hierarchy(document.content, document.doc_id)
            for item in hierarchy:
                chunks.append(
                    Chunk(
                        chunk_id=item["child_id"],
                        doc_id=document.doc_id,
                        text=item["child_text"],  # BM25 indexes the child text for keyword precision
                        chunk_index=item["chunk_index"],
                        parent_id=item["parent_id"],
                        parent_text=item["parent_text"],
                        # We can store parent text/id in attributes or metadata if needed, 
                        # but for BM25, scoring happens on child text, and we swap to parent text when creating Evidence.
                    )
                )
        return chunks

    # Fallback for standard chunkers (Recursive, Fixed, etc.)
    for document in documents:
        text_chunks = use_chunker.split(document.content)

        for index, text in enumerate(text_chunks):
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:{index}",
                    doc_id=document.doc_id,
                    text=text,
                    chunk_index=index,
                )
            )

    return chunks


class KeywordSearch(SearchStrategyBase):
    """Simple keyword-based search over document chunks."""

    def __init__(self, chunker: ChunkingStrategyBase | None = None):
        self.chunker = chunker or WholeDocumentChunking()

    def search(
        self,
        query: str,
        documents: list[Document],
        limit: int = 4,
        scope: str = "global",
        doc_id: str | None = None,
) -> list[Evidence]:

        if limit < 1:
            raise ValueError("limit must be positive")

        if scope not in {"global", "document"}:
                raise ValueError("scope must be 'global' or 'document'")

        if scope == "document":
            if not doc_id:
                raise ValueError("doc_id is required for document scope")

            documents = [doc for doc in documents if doc.doc_id == doc_id]

            if not documents:
                raise ValueError(f"Document not found: {doc_id}")

        terms = set(re.findall(r"\w+", query.lower()))

        chunks = build_chunks(documents, self.chunker)

        ranked = []

        for chunk in chunks:
            chunk_terms = set(re.findall(r"\w+", chunk.text.lower()))

            overlap = len(terms & chunk_terms)

            if overlap > 0:
                ranked.append(
                    Evidence(
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        score=float(overlap),
                    )
                )

        ranked.sort(key=lambda item: item.score, reverse=True)

        return ranked[:limit]

class BM25Search(SearchStrategyBase):
    """Sparse lexical retrieval using the BM25 ranking algorithm."""

    def __init__(self, chunker: ChunkingStrategyBase | None = None):
        self.chunker = chunker or WholeDocumentChunking()

    def search(
        self,
        query: str,
        documents: list[Document],
        limit: int = 4,
        scope: str = "global",
        doc_id: str | None = None,
) -> list[Evidence]:

        if limit < 1:
            raise ValueError("limit must be positive")

        if scope not in {"global", "document"}:
            raise ValueError("scope must be 'global' or 'document'")

        if scope == "document":
            if not doc_id:
                raise ValueError("doc_id is required for document scope")

            documents = [doc for doc in documents if doc.doc_id == doc_id]

            if not documents:
                raise ValueError(f"Document not found: {doc_id}")

        chunks = build_chunks(documents, self.chunker)

        if not chunks:
            return []

        # Tokenize each chunk to create the BM25 corpus.
        tokenized_chunks = [
            re.findall(r"\w+", chunk.text.lower())
            for chunk in chunks
        ]

        # Build the BM25 index over the chunks.
        bm25 = BM25Okapi(tokenized_chunks)

        # Tokenize the incoming question.
        query_tokens = re.findall(r"\w+", query.lower())

        if not query_tokens:
            return []

        # Calculate BM25 relevance scores.
        scores = bm25.get_scores(query_tokens)

        ranked = []

        for chunk, score in zip(chunks, scores):
            if score > 0:
                # If using parent-child, resolve the parent text for the evidence
                evidence_text = chunk.text
                if hasattr(self.chunker, "get_parent_text"):
                    evidence_text = self.chunker.get_parent_text(chunk.chunk_id)

                ranked.append(
                    Evidence(
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        text=evidence_text,  # Returns Parent text for LLM consistency!
                        score=float(score),
                    )
                )

        ranked.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return ranked[:limit]


class HybridSearch(SearchStrategyBase):
    """Combines multiple retrieval strategies using RRF."""

    def __init__(
        self,
        dense_search: SearchStrategyBase,
        sparse_search: SearchStrategyBase,
        rrf_k: int = 60,
    ):
        if rrf_k < 1:
            raise ValueError("rrf_k must be positive")

        self.dense_search = dense_search
        self.sparse_search = sparse_search
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        documents: list[Document],
        limit: int = 4,
        scope: str = "global",
        doc_id: str | None = None,
) -> list[Evidence]:

        if limit < 1:
            raise ValueError("limit must be positive")

        # Retrieve candidates from both search systems.
        dense_results = self.dense_search.search(
            query=query,
            documents=documents,
            limit=limit,
            scope=scope,
            doc_id=doc_id,
        )

        sparse_results = self.sparse_search.search(
            query=query,
            documents=documents,
            limit=limit,
            scope=scope,
            doc_id=doc_id,
        )

        # Store the fused score for each unique chunk.
        fused_scores = {}
        evidence_by_chunk = {}

        # Apply Reciprocal Rank Fusion to dense results.
        for rank, evidence in enumerate(dense_results, start=1):

            chunk_id = evidence.chunk_id

            if chunk_id is None:
                continue

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

            evidence_by_chunk[chunk_id] = evidence

        # Apply Reciprocal Rank Fusion to sparse results.
        for rank, evidence in enumerate(sparse_results, start=1):

            chunk_id = evidence.chunk_id

            if chunk_id is None:
                continue

            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )

            evidence_by_chunk[chunk_id] = evidence

        # Sort candidates by their combined RRF score.
        ranked_ids = sorted(
            fused_scores,
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True,
        )

        results = []

        for chunk_id in ranked_ids[:limit]:

            evidence = evidence_by_chunk[chunk_id]

            results.append(
                Evidence(
                    doc_id=evidence.doc_id,
                    chunk_id=evidence.chunk_id,
                    text=evidence.text,
                    score=fused_scores[chunk_id],
                )
            )

        return results


class LimitedSearch(SearchStrategyBase):
    """Restricts the number of returned search results."""

    def __init__(self, base: SearchStrategyBase, limit: int):
        if limit < 1:
            raise ValueError("limit must be positive")

        self.base = base
        self.limit = limit

    def index_documents(self, documents: list[Document]) -> None:
        """Forward indexing to the underlying search strategy."""

        if not hasattr(self.base, "index_documents"):
            raise NotImplementedError(
                "The configured search strategy does not support indexing"
            )

        self.base.index_documents(documents)

    def search(
        self,
        query: str,
        documents: list[Document],
        limit: int = 4,
        scope: str = "global",
        doc_id: str | None = None,
) -> list[Evidence]:

        return self.base.search(
            query=query,
            documents=documents,
            limit=self.limit,
            scope=scope,
            doc_id=doc_id,
        )


class ChromaSearch(SearchStrategyBase):
    """Dense vector search using ChromaDB and Sentence Transformers."""

    def __init__(
        self,
        path: str | None = None,
        embedder=None,
        collection_name: str = "finance_sample",
        chunker: ChunkingStrategyBase | None = None,
    ):
        import chromadb

        self.client = chromadb.PersistentClient(
            path=path or os.getenv(
                "CHROMA_PATH",
                "data/processed/chroma",
            )
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "COLLECTION_INITIALIZED | name=%s | metadata=%s | count=%s",
            self.collection.name,
            self.collection.metadata,
            self.collection.count(),
        )
        print("Chroma collection metadata:", self.collection.metadata)

        # Keep the existing default for backward compatibility.
        # The factory will inject SentenceTransformerEmbedder.
        self.embedder = embedder or HashingEmbedder()

        # The factory will inject RecursiveChunking.
        self.chunker = chunker or WholeDocumentChunking()

    def index_documents(self, documents: list[Document]) -> None:
        """Chunk, embed and store document chunks in ChromaDB."""

        chunks = build_chunks(
            documents=documents,
            chunker=self.chunker,
        )

        if not chunks:
            return

        logger.info(
            "INDEXING_STARTED | documents=%d | chunks=%d | embedder=%s",
            len(documents),
            len(chunks),
            type(self.embedder).__name__,
        )

        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedder.encode(texts)

        logger.info(
            "EMBEDDINGS_CREATED | count=%d | dimensions=%d",
            len(embeddings),
            len(embeddings[0]) if embeddings else 0,
        )

        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],

            documents=texts,

            metadatas=[
            {
                "doc_id": chunk.doc_id,
                "parent_id": chunk.parent_id or "",
                "parent_text": chunk.parent_text or chunk.text,
                "chunk_index": chunk.chunk_index,
            }
            for chunk in chunks
            ],

        embeddings=embeddings,
         )
        logger.info(
            "INDEXING_COMPLETED | collection=%s | stored_count=%d",
            self.collection.name,
            self.collection.count(),
        )

    def search(
        self,
        query: str,
        documents: list[Document],
        limit: int = 4,
        scope: str = "global",
        doc_id: str | None = None,
) -> list[Evidence]:
        """Search previously indexed documents."""

        if limit < 1:
            raise ValueError("limit must be positive")

        if scope not in {"global", "document"}:
            raise ValueError("scope must be 'global' or 'document'")

        if scope == "document":
            if not doc_id:
                raise ValueError("doc_id is required for document scope")

            if not any(doc.doc_id == doc_id for doc in documents):
                raise ValueError(f"Document not found: {doc_id}")

        # No indexing here. Documents must be indexed beforehand.
        if scope == "document":
            matching_records = self.collection.get(
                where={"doc_id": doc_id},
                include=[],
            )

            if not matching_records["ids"]:
                return []

            count = len(matching_records)

        else:
            count = self.collection.count()

        if count == 0:
            return []

        logger.info(
            "QUERY_EMBEDDING_STARTED | embedder=%s | scope=%s | doc_id=%s",
            type(self.embedder).__name__,
            scope,
            doc_id,
        )

        if hasattr(self.embedder, "embed_query"):
            query_embedding = [self.embedder.embed_query(query)]
        else:
            query_embedding = self.embedder.encode([query])

        logger.info(
            "QUERY_EMBEDDING_COMPLETED | dimensions=%d",
            len(query_embedding[0]),
        )
        query_kwargs = {
            "query_embeddings": query_embedding,
            "n_results": min(limit, count),
            "include": ["documents", "metadatas", "distances"],
        }

        if scope == "document":
            query_kwargs["where"] = {"doc_id": doc_id}

        result = self.collection.query(**query_kwargs)

        logger.info(
            "SEARCH_EXECUTED | query=%r | scope=%s | doc_id=%s | requested=%d",
            query,
            scope,
            doc_id,
            query_kwargs["n_results"],
        )

        retrieved = []

        for chunk_id, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):

            # Key Parent-Child Resolution Step:
            # If parent_text exists in metadata, swap child text for the parent text block!
            context_text = metadata.get("parent_text", text)
            retrieved.append(
                Evidence(
                    doc_id=metadata["doc_id"],
                    chunk_id=chunk_id,
                    text=context_text,
                    score=1.0 - float(distance),
                )
            )

        for item in retrieved:
            logger.info(
                "RETRIEVED_CHUNK | doc_id=%s | chunk_id=%s | distance=%s | similarity=%s",
                item.doc_id,
                item.chunk_id,
                round(1.0 - item.score, 6),
                round(item.score, 6),
            )

            logger.info(
                "SEARCH_COMPLETED | returned_chunks=%d",
                len(retrieved),
            )
        return retrieved
