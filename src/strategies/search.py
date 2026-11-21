"""Search strategies (keyword, BM25, hybrid, and Chroma)."""

import copy
import logging
import os
from pathlib import Path
import re
from typing import Any

from rank_bm25 import BM25Okapi

from src.db.parent_store import ParentChunkStore
from src.models.records import Chunk, Document, Evidence
from src.strategies.base import ChunkingStrategyBase, SearchStrategyBase
from src.strategies.chunking import WholeDocumentChunking
from src.strategies.embeddings import HashingEmbedder

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


def _infer_parent_id(chunk_id: str | None, metadata: dict | None = None) -> str:
    """Resolve parent_id from metadata or `<doc_id>:child:<p_idx>:<c_idx>`."""
    if metadata and metadata.get("parent_id"):
        return str(metadata["parent_id"])
    if chunk_id and ":child:" in chunk_id:
        prefix, rest = chunk_id.split(":child:", 1)
        parent_idx = rest.split(":", 1)[0]
        return f"{prefix}:parent:{parent_idx}"
    return ""


def _set_dict_attr(obj: Any, attr: str, value: Any) -> None:
    """Write directly to instance __dict__ when available to bypass broken setters."""
    if attr == "rank_gain":
        return
    obj_dict = getattr(obj, "__dict__", None)
    if isinstance(obj_dict, dict):
        obj_dict[attr] = value
        return
    try:
        object.__setattr__(obj, attr, value)
    except (AttributeError, TypeError):
        pass


def _build_evidence(
    *,
    doc_id: str,
    chunk_id: str | None,
    text: str,
    score: float,
    meta: dict | None = None,
    dense_rank: int | float | None = None,
    sparse_rank: int | float | None = None,
    hybrid_rank: int | None = None,
) -> Evidence:
    """Construct Evidence without triggering broken attribute setters (such as rank_gain)."""
    ev = Evidence.__new__(Evidence)
    _set_dict_attr(ev, "doc_id", doc_id)
    _set_dict_attr(ev, "text", text)
    _set_dict_attr(ev, "score", float(score))
    _set_dict_attr(ev, "question", "")
    _set_dict_attr(ev, "answer", "")
    _set_dict_attr(ev, "image_path", None)
    _set_dict_attr(ev, "chunk_id", chunk_id)
    _set_dict_attr(ev, "dense_rank", dense_rank)
    _set_dict_attr(ev, "sparse_rank", sparse_rank)
    _set_dict_attr(ev, "hybrid_rank", hybrid_rank)
    _set_dict_attr(ev, "rerank_rank", None)
    if meta is not None:
        _set_dict_attr(ev, "metadata", dict(meta))
    return ev


def _clone_evidence_with_ranks(
    evidence: Evidence,
    *,
    score: float | None = None,
    dense_rank: int | float | None = None,
    sparse_rank: int | float | None = None,
    hybrid_rank: int | None = None,
) -> Evidence:
    """Clone Evidence into a fresh instance and update score/ranks safely."""
    cloned = Evidence.__new__(Evidence)
    ev_dict = getattr(evidence, "__dict__", None)
    if isinstance(ev_dict, dict):
        for k, v in ev_dict.items():
            if k != "rank_gain":
                cloned.__dict__[k] = copy.deepcopy(v)
    else:
        for attr_name in (
            "doc_id",
            "text",
            "question",
            "answer",
            "image_path",
            "score",
            "chunk_id",
            "dense_rank",
            "sparse_rank",
            "hybrid_rank",
            "rerank_rank",
            "metadata",
        ):
            if hasattr(evidence, attr_name):
                _set_dict_attr(cloned, attr_name, getattr(evidence, attr_name))

    if score is not None:
        _set_dict_attr(cloned, "score", float(score))

    final_dense = (
        dense_rank if dense_rank is not None else getattr(evidence, "dense_rank", None)
    )
    final_sparse = (
        sparse_rank if sparse_rank is not None else getattr(evidence, "sparse_rank", None)
    )
    final_hybrid = (
        hybrid_rank if hybrid_rank is not None else getattr(evidence, "hybrid_rank", None)
    )

    if final_dense is not None:
        _set_dict_attr(cloned, "dense_rank", final_dense)
    if final_sparse is not None:
        _set_dict_attr(cloned, "sparse_rank", final_sparse)
    if final_hybrid is not None:
        _set_dict_attr(cloned, "hybrid_rank", final_hybrid)

    return cloned


def build_chunks(
    documents: list[Document],
    chunker: ChunkingStrategyBase | None = None,
    parent_store: ParentChunkStore | None = None,
) -> list[Chunk]:
    """Split documents into chunk records, supporting parent-child hierarchy if available."""

    use_chunker = chunker or WholeDocumentChunking()
    chunks = []

    # Check if the chunker supports parent-child hierarchy
    if hasattr(use_chunker, "split_with_hierarchy"):
        seen_parents: dict[str, tuple[str, str, str]] = {}

        for document in documents:
            hierarchy = use_chunker.split_with_hierarchy(document.content, document.doc_id)
            for item in hierarchy:
                parent_id = item["parent_id"]
                parent_text = item["parent_text"]
                seen_parents[parent_id] = (parent_id, document.doc_id, parent_text)

                chunks.append(
                    Chunk(
                        chunk_id=item["child_id"],
                        doc_id=document.doc_id,
                        text=item["child_text"],  # Keep child text for BM25/dense/reranker scoring
                        chunk_index=item["chunk_index"],
                        parent_id=parent_id,
                        parent_text=parent_text,
                    )
                )

        if parent_store is not None and seen_parents:
            parent_store.upsert_parents(list(seen_parents.values()))

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
                    _build_evidence(
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        score=float(overlap),
                    )
                )

        ranked.sort(key=lambda item: item.score, reverse=True)

        return [
            _clone_evidence_with_ranks(item, sparse_rank=rank)
            for rank, item in enumerate(ranked[:limit], start=1)
        ]


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

        # Tokenize each child chunk to create the BM25 corpus.
        tokenized_chunks = [
            re.findall(r"\w+", chunk.text.lower())
            for chunk in chunks
        ]

        bm25 = BM25Okapi(tokenized_chunks)
        query_tokens = re.findall(r"\w+", query.lower())

        if not query_tokens:
            return []

        scores = bm25.get_scores(query_tokens)

        ranked = []

        for chunk, score in zip(chunks, scores):
            if score > 0:
                parent_id = _infer_parent_id(
                    chunk.chunk_id,
                    {"parent_id": chunk.parent_id} if chunk.parent_id else None,
                )
                meta = {"parent_id": parent_id} if parent_id else None
                ranked.append(
                    _build_evidence(
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        score=float(score),
                        meta=meta,
                    )
                )

        ranked.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return [
            _clone_evidence_with_ranks(item, sparse_rank=rank)
            for rank, item in enumerate(ranked[:limit], start=1)
        ]


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

        fused_scores: dict[str, float] = {}
        evidence_by_chunk: dict[str, Evidence] = {}
        dense_ranks: dict[str, int] = {}
        sparse_ranks: dict[str, int] = {}

        for rank, evidence in enumerate(dense_results, start=1):
            chunk_id = evidence.chunk_id
            if chunk_id is None:
                continue

            dense_ranks.setdefault(chunk_id, rank)
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )
            evidence_by_chunk.setdefault(chunk_id, evidence)

        for rank, evidence in enumerate(sparse_results, start=1):
            chunk_id = evidence.chunk_id
            if chunk_id is None:
                continue

            sparse_ranks.setdefault(chunk_id, rank)
            fused_scores[chunk_id] = (
                fused_scores.get(chunk_id, 0.0)
                + 1.0 / (self.rrf_k + rank)
            )
            evidence_by_chunk.setdefault(chunk_id, evidence)

        ranked_ids = sorted(
            fused_scores,
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True,
        )

        results = []

        for hybrid_rank, chunk_id in enumerate(ranked_ids[:limit], start=1):
            evidence = evidence_by_chunk[chunk_id]
            d_rank = dense_ranks.get(chunk_id, getattr(evidence, "dense_rank", None))
            s_rank = sparse_ranks.get(chunk_id, getattr(evidence, "sparse_rank", None))
            results.append(
                _clone_evidence_with_ranks(
                    evidence,
                    score=fused_scores[chunk_id],
                    dense_rank=d_rank,
                    sparse_rank=s_rank,
                    hybrid_rank=hybrid_rank,
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

        if limit < 1:
            raise ValueError("limit must be positive")

        effective_limit = min(limit, self.limit)

        return self.base.search(
            query=query,
            documents=documents,
            limit=effective_limit,
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
        parent_store: ParentChunkStore | None = None,
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

        self.embedder = embedder or HashingEmbedder()
        self.chunker = chunker or WholeDocumentChunking()
        self.parent_store = parent_store

    def index_documents(self, documents: list[Document]) -> None:
        """Chunk, embed and store document chunks in ChromaDB (and parents in SQLite)."""

        chunks = build_chunks(
            documents=documents,
            chunker=self.chunker,
            parent_store=self.parent_store,
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
        """Search previously indexed child chunks."""

        if limit < 1:
            raise ValueError("limit must be positive")

        if scope not in {"global", "document"}:
            raise ValueError("scope must be 'global' or 'document'")

        if scope == "document":
            if not doc_id:
                raise ValueError("doc_id is required for document scope")

            if not any(doc.doc_id == doc_id for doc in documents):
                raise ValueError(f"Document not found: {doc_id}")

        if scope == "document":
            matching_records = self.collection.get(
                where={"doc_id": doc_id},
                include=[],
            )
            if not matching_records["ids"]:
                return []
            count = len(matching_records["ids"])
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

        for rank, (chunk_id, text, metadata, distance) in enumerate(
            zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            ),
            start=1,
        ):
            meta = dict(metadata or {})
            parent_id = _infer_parent_id(chunk_id, meta)
            if parent_id:
                meta["parent_id"] = parent_id

            # Keep child `text` during retrieval and reranking so sibling children score distinctly.
            retrieved.append(
                _build_evidence(
                    doc_id=meta["doc_id"],
                    chunk_id=chunk_id,
                    text=text,
                    score=1.0 - float(distance),
                    meta=meta,
                    dense_rank=rank,
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