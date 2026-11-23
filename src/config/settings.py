"""Environment-driven configuration for strategy selection."""

import os
from dataclasses import dataclass


def _load_local_env(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class AppConfig:
    search_strategy: str = "chroma"
    reranker_model: str = "BAAI/bge-reranker-base"
    rag_strategy: str = "auto"
    prompt_strategy: str = "simple"
    answer_strategy: str = "dataset"
    use_multi_agent_graph: bool = False
    agent_max_attempts: int = 2
    agent_top_k: int = 5
    memory_strategy: str = "conversation"
    evaluator_strategy: str = "simple"
    chunking_strategy: str = "whole"
    embedder_strategy: str = "hashing"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_path: str = "data/processed/chroma"
    chroma_collection: str = "finance_sample"
    metrics_db_path: str = "data/processed/metrics.sqlite3"
    chunk_size: int = 800
    chunk_overlap: int = 100
    retrieval_limit: int = 4

    @classmethod
    def from_env(cls) -> "AppConfig":
        _load_local_env()
        return cls(
            search_strategy=os.getenv("SEARCH_STRATEGY", cls.search_strategy),
            rag_strategy=os.getenv("RAG_STRATEGY", cls.rag_strategy),
            prompt_strategy=os.getenv("PROMPT_STRATEGY", cls.prompt_strategy),
            answer_strategy=os.getenv("ANSWER_STRATEGY", cls.answer_strategy),
            use_multi_agent_graph=_get_bool_env("USE_MULTI_AGENT_GRAPH", cls.use_multi_agent_graph),
            agent_max_attempts=int(os.getenv("AGENT_MAX_ATTEMPTS", str(cls.agent_max_attempts))),
            agent_top_k=int(os.getenv("AGENT_TOP_K", str(cls.agent_top_k))),
            memory_strategy=os.getenv("MEMORY_STRATEGY", cls.memory_strategy),
            evaluator_strategy=os.getenv("EVALUATOR_STRATEGY", cls.evaluator_strategy),
            chunking_strategy=os.getenv("CHUNKING_STRATEGY", cls.chunking_strategy),
            embedder_strategy=os.getenv("EMBEDDER_STRATEGY", cls.embedder_strategy),
            embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", cls.embedding_model_name),
            chroma_path=os.getenv("CHROMA_PATH", cls.chroma_path),
            chroma_collection=os.getenv("CHROMA_COLLECTION", cls.chroma_collection),
            metrics_db_path=os.getenv("METRICS_DB_PATH", cls.metrics_db_path),
            chunk_size=int(os.getenv("CHUNK_SIZE", str(cls.chunk_size))),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", str(cls.chunk_overlap))),
            retrieval_limit=int(os.getenv("RETRIEVAL_LIMIT", str(cls.retrieval_limit))),
        )
