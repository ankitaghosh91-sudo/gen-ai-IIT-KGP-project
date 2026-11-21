"""Build pipeline strategies from config values."""

from dataclasses import is_dataclass, replace
import logging
from typing import Any, Callable

from src.agents.generator import make_answer_strategy_generator
from src.agents.graph import build_rag_graph
from src.agents.retriever import (
    make_rag_strategy_retriever,
    make_search_strategy_retriever,
)
from src.config import AppConfig
from src.db.parent_store import ParentChunkStore
from src.models.records import Document
from src.pipeline.runner import PipelineStrategies
from src.strategies import (
    AgenticRAG,
    AutoRAG,
    BGEAPIEmbedder,
    BGEReranker,
    BM25Search,
    ChromaSearch,
    CombinedRAG,
    ConversationMemory,
    CriticAnswer,
    DatasetAnswer,
    DecompositionPrompt,
    FewShotPrompt,
    FixedSizeChunking,
    HashingEmbedder,
    HybridSearch,
    KeywordSearch,
    LimitedSearch,
    LLMAnswer,
    MultiStepRAG,
    NoMemory,
    ParagraphChunking,
    ParentChildChunker,
    RecursiveChunking,
    SearchRerankerPipeline,
    SentenceTransformerEmbedder,
    SimpleEvaluator,
    SimplePrompt,
    SimpleRAG,
    VisualRAG,
    VotingAnswer,
    VotingEvaluator,
    WholeDocumentChunking,
    ZeroShotPrompt,
)
from src.strategies.base import RAGStrategyBase, SearchStrategyBase

logger = logging.getLogger(__name__)


def _get_field(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _infer_parent_id(item: Any) -> str:
    meta = _get_field(item, "metadata", None)
    if isinstance(meta, dict) and meta.get("parent_id"):
        return str(meta["parent_id"])
    pid = _get_field(item, "parent_id", None)
    if pid:
        return str(pid)
    chunk_id = str(_get_field(item, "chunk_id", "") or "")
    if ":child:" in chunk_id:
        prefix, rest = chunk_id.split(":child:", 1)
        parent_idx = rest.split(":", 1)[0]
        return f"{prefix}:parent:{parent_idx}"
    return ""


class _LoggingRerankerWrapper:
    """Logs child evidence + parent metadata going into and coming out of the reranker."""

    def __init__(
        self,
        reranker: Any,
        parent_store: ParentChunkStore | None = None,
        sample_log_count: int = 5,
    ) -> None:
        self.reranker = reranker
        self.parent_store = parent_store
        self.sample_log_count = sample_log_count

    def rerank(self, query: str, results: list, top_k: int = 5) -> list:
        parent_ids = [_infer_parent_id(item) for item in results]
        unique_parent_ids = list(dict.fromkeys(pid for pid in parent_ids if pid))

        stored_parents: dict[str, str] = {}
        if self.parent_store is not None and unique_parent_ids:
            stored_parents = self.parent_store.get_parent_texts(unique_parent_ids)

        missing_parents = [pid for pid in unique_parent_ids if pid not in stored_parents]
        logger.info(
            "RERANKER_TRACE_IN | query=%r | candidates=%d | top_k=%d | unique_parents=%d | found_in_store=%d | missing_in_store=%s",
            query,
            len(results),
            top_k,
            len(unique_parent_ids),
            len(stored_parents),
            missing_parents[:10],
        )

        for idx, item in enumerate(results[: self.sample_log_count], start=1):
            chunk_id = _get_field(item, "chunk_id", "")
            parent_id = _infer_parent_id(item)
            child_text = str(_get_field(item, "text", "") or "")
            parent_text = stored_parents.get(parent_id, "")
            logger.info(
                "RERANKER_CHILD_EVIDENCE | idx=%d | chunk_id=%s | parent_id=%s | child_len=%d | parent_len=%d | is_same_as_parent=%s | child_preview=%r",
                idx,
                chunk_id,
                parent_id,
                len(child_text),
                len(parent_text),
                bool(parent_text and child_text == parent_text),
                child_text[:120],
            )

        reranked = self.reranker.rerank(query=query, results=results, top_k=top_k)

        for idx, item in enumerate(reranked[: self.sample_log_count], start=1):
            chunk_id = _get_field(item, "chunk_id", "")
            parent_id = _infer_parent_id(item)
            score = _get_field(item, "score", None)
            child_text = str(_get_field(item, "text", "") or "")
            logger.info(
                "RERANKER_TRACE_OUT | rank=%d | chunk_id=%s | parent_id=%s | rerank_score=%s | child_len=%d",
                idx,
                chunk_id,
                parent_id,
                score,
                len(child_text),
            )

        return reranked


def _build_chunker(config: AppConfig):
    if config.chunking_strategy == "whole":
        return WholeDocumentChunking()
    if config.chunking_strategy == "fixed":
        return FixedSizeChunking(size=config.chunk_size, overlap=config.chunk_overlap)
    if config.chunking_strategy == "paragraph":
        return ParagraphChunking()
    if config.chunking_strategy == "recursive":
        return RecursiveChunking(
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
    if config.chunking_strategy == "parent_child":
        child_size = config.chunk_size
        parent_size = max(1200, child_size * 4)
        parent_overlap = min(150, max(0, parent_size // 8))
        child_overlap = min(config.chunk_overlap, max(0, child_size // 4))
        logger.info(
            "PARENT_CHUNKER_CONFIG | child_size=%d | child_overlap=%d | parent_size=%d | parent_overlap=%d",
            child_size,
            child_overlap,
            parent_size,
            parent_overlap,
        )
        return ParentChildChunker(
            parent_size=parent_size,
            parent_overlap=parent_overlap,
            child_size=child_size,
            child_overlap=child_overlap,
        )
    raise ValueError(f"Unknown chunking strategy: {config.chunking_strategy}")


def _build_parent_store(config: AppConfig, chunker: Any) -> ParentChunkStore | None:
    if config.chunking_strategy == "parent_child" or hasattr(chunker, "split_with_hierarchy"):
        db_path = getattr(config, "parent_store_path", "data/processed/parent_chunks.db")
        store = ParentChunkStore(db_path=db_path)
        logger.info("PARENT_STORE_INITIALIZED | db_path=%s", store.db_path)
        return store
    return None


def _build_embedder(config: AppConfig):
    if config.embedder_strategy == "hashing":
        logger.info(
            "EMBEDDER_INITIALIZED | strategy=%s | model=%s",
            config.embedder_strategy,
            config.embedding_model_name,
        )
        return HashingEmbedder()
    if config.embedder_strategy == "sentence_transformer":
        logger.info(
            "EMBEDDER_INITIALIZED | strategy=%s | model=%s",
            config.embedder_strategy,
            config.embedding_model_name,
        )
        return SentenceTransformerEmbedder(config.embedding_model_name)
    if config.embedder_strategy == "bge_api":
        logger.info(
            "EMBEDDER_INITIALIZED | strategy=%s | model=%s",
            config.embedder_strategy,
            config.embedding_model_name,
        )
        return BGEAPIEmbedder(
            model_name=config.embedding_model_name,
        )
    raise ValueError(f"Unknown embedder strategy: {config.embedder_strategy}")


def _build_search(config: AppConfig):
    chunker = _build_chunker(config)
    embedder = _build_embedder(config)
    parent_store = _build_parent_store(config, chunker)

    if config.search_strategy == "keyword":
        base = KeywordSearch(chunker=chunker)
    elif config.search_strategy == "bm25":
        base = BM25Search(chunker=chunker)
    elif config.search_strategy == "chroma":
        base = ChromaSearch(
            path=config.chroma_path,
            collection_name=config.chroma_collection,
            embedder=embedder,
            chunker=chunker,
            parent_store=parent_store,
        )
    elif config.search_strategy == "hybrid":
        chroma_base = ChromaSearch(
            path=config.chroma_path,
            collection_name=config.chroma_collection,
            embedder=embedder,
            chunker=chunker,
            parent_store=parent_store,
        )
        bm25_base = BM25Search(chunker=chunker)
        base = HybridSearch(
            dense_search=chroma_base,
            sparse_search=bm25_base,
            rrf_k=60,
        )
    elif config.search_strategy in ("hybrid_reranker", "hybrid-reranker"):
        chroma_base = ChromaSearch(
            path=config.chroma_path,
            collection_name=config.chroma_collection,
            embedder=embedder,
            chunker=chunker,
            parent_store=parent_store,
        )
        bm25_base = BM25Search(chunker=chunker)
        hybrid_base = HybridSearch(
            dense_search=chroma_base,
            sparse_search=bm25_base,
            rrf_k=60,
        )
        limited_base = LimitedSearch(hybrid_base, config.retrieval_limit)

        reranker_model = BGEReranker(
            model_name=getattr(config, "reranker_model", "BAAI/bge-reranker-base")
        )
        logged_reranker = _LoggingRerankerWrapper(
            reranker=reranker_model,
            parent_store=parent_store,
        )

        base = SearchRerankerPipeline(
            base_search=limited_base,
            reranker=logged_reranker,
            top_k=config.retrieval_limit,
            candidate_pool_size=config.retrieval_limit,
        )
        return base
    else:
        raise ValueError(f"Unknown search strategy: {config.search_strategy}")

    return LimitedSearch(base, config.retrieval_limit)


def _build_rag(config: AppConfig, search):
    if config.rag_strategy == "simple":
        return SimpleRAG(search)
    if config.rag_strategy == "multistep":
        return MultiStepRAG(search)
    if config.rag_strategy == "agentic":
        return AgenticRAG(search)
    if config.rag_strategy == "visual":
        return VisualRAG()
    if config.rag_strategy == "combined":
        return CombinedRAG(text=SimpleRAG(search), visual=VisualRAG())
    if config.rag_strategy == "auto":
        return AutoRAG(text=SimpleRAG(search), visual=VisualRAG())
    raise ValueError(f"Unknown RAG strategy: {config.rag_strategy}")


def _build_prompt(config: AppConfig):
    if config.prompt_strategy == "simple":
        return SimplePrompt()
    if config.prompt_strategy == "zero_shot":
        return ZeroShotPrompt()
    if config.prompt_strategy == "few_shot":
        return FewShotPrompt()
    if config.prompt_strategy == "decomposition":
        return DecompositionPrompt()
    raise ValueError(f"Unknown prompt strategy: {config.prompt_strategy}")


def _build_answer(config: AppConfig, model_callable: Callable[[str], dict] | None = None):
    if config.answer_strategy == "dataset":
        return DatasetAnswer()
    if config.answer_strategy == "critic":
        return CriticAnswer(DatasetAnswer())
    if config.answer_strategy == "voting":
        return VotingAnswer([DatasetAnswer(), CriticAnswer(DatasetAnswer())])
    if config.answer_strategy == "llm":
        if model_callable is None:
            raise ValueError("ANSWER_STRATEGY=llm requires a model callable.")
        return LLMAnswer(model_callable)
    raise ValueError(f"Unknown answer strategy: {config.answer_strategy}")


def _build_memory(config: AppConfig):
    if config.memory_strategy == "conversation":
        return ConversationMemory()
    if config.memory_strategy == "none":
        return NoMemory()
    raise ValueError(f"Unknown memory strategy: {config.memory_strategy}")


def _build_evaluator(config: AppConfig):
    if config.evaluator_strategy == "simple":
        return SimpleEvaluator()
    if config.evaluator_strategy == "voting":
        left = SimpleEvaluator()
        right = SimpleEvaluator()
        return VotingEvaluator(
            [
                lambda answer, evidence: left.evaluate(answer, evidence),
                lambda answer, evidence: right.evaluate(answer, evidence),
            ]
        )
    if config.evaluator_strategy == "judge":
        raise ValueError("EVALUATOR_STRATEGY=judge needs a custom judge callable.")
    raise ValueError(f"Unknown evaluator strategy: {config.evaluator_strategy}")


def build_strategies_from_config(
    config: AppConfig,
    model_callable: Callable[[str], dict] | None = None,
) -> PipelineStrategies:
    search = _build_search(config)
    return PipelineStrategies(
        rag=_build_rag(config, search),
        prompt=_build_prompt(config),
        answerer=_build_answer(config, model_callable=model_callable),
        memory=_build_memory(config),
        evaluator=_build_evaluator(config),
    )


def build_agent_rag_graph(
    config: AppConfig,
    rag_or_search,
    prompt_strategy,
    documents: list[Document],
    model_callable: Callable[[str], dict] | None = None,
    rerank_fn=None,
    image=None,
):
    """Wires existing retrieval + prompt + LLMAnswer strategies into the 6-agent LangGraph."""
    if isinstance(rag_or_search, RAGStrategyBase):
        search_fn = make_rag_strategy_retriever(
            rag_strategy=rag_or_search,
            documents=documents,
            image=image,
        )
    elif isinstance(rag_or_search, SearchStrategyBase):
        search_fn = make_search_strategy_retriever(
            search_strategy=rag_or_search,
            documents=documents,
            limit=max(getattr(config, "agent_top_k", 5) * 2, 8),
        )
    elif hasattr(rag_or_search, "retrieve"):
        search_fn = make_rag_strategy_retriever(
            rag_strategy=rag_or_search,
            documents=documents,
            image=image,
        )
    else:
        search_fn = make_search_strategy_retriever(
            search_strategy=rag_or_search,
            documents=documents,
            limit=max(getattr(config, "agent_top_k", 5) * 2, 8),
        )

    answer_strategy = _build_answer(config, model_callable=model_callable)
    generate_fn = make_answer_strategy_generator(
        answer_strategy=answer_strategy,
        prompt_strategy=prompt_strategy,
        documents=documents,
        max_docs=getattr(config, "agent_top_k", 5),
    )

    return build_rag_graph(
        search_fn=search_fn,
        rerank_fn=rerank_fn,
        top_k=getattr(config, "agent_top_k", 5),
        generate_fn=generate_fn,
    )