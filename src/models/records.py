"""Core record shapes used in the app and pipeline."""

from dataclasses import dataclass


@dataclass
class QAPair:
    question: str
    answer: str


@dataclass
class Document:
    doc_id: str
    content: str
    qa_pairs: list[QAPair]
    image_path: str | None = None


@dataclass
class Evidence:
    doc_id: str
    text: str
    question: str = ""
    answer: str = ""
    image_path: str | None = None
    score: float = 0.0
    chunk_id: str | None = None
    dense_rank: int | float | None = None
    sparse_rank: int | float | None = None
    hybrid_rank: int | None = None
    rerank_rank: int | None = None
    rank_gain: int | None = None

    @property
    def pre_rerank_rank(self) -> int | float | None:
        if self.hybrid_rank is not None:
            return self.hybrid_rank
        candidates = [r for r in (self.dense_rank, self.sparse_rank) if r is not None]
        return min(candidates) if candidates else None

    @property
    def rank_gain(self) -> int | None:
        if self.pre_rerank_rank is None or self.rerank_rank is None:
            return None
        return int(self.pre_rerank_rank) - int(self.rerank_rank)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    parent_id: str | None = None
    parent_text: str | None = None