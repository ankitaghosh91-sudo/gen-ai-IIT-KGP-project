from collections.abc import Callable
import inspect
import logging
from typing import Any

from src.agents.state import AgentRAGState
from src.models.records import Document, Evidence

logger = logging.getLogger(__name__)

RetrieverFn = Callable[[str, list[str] | None, str], list[Evidence | dict[str, Any]]]


def _derive_parent_id(chunk_id: str, metadata: dict[str, Any]) -> str | None:
    if metadata.get("parent_id"):
        return str(metadata["parent_id"])
    parts = chunk_id.split(":")
    if len(parts) >= 3 and parts[1] == "child":
        return f"{parts[0]}:parent:{parts[2]}"
    return None


def _normalize_doc_id(doc_id: Any) -> str | None:
    if doc_id is None:
        return None
    raw = str(doc_id).strip()
    if not raw:
        return None
    return raw


def _matches_scope(item_doc_id: Any, allowed_ids: set[str]) -> bool:
    norm = _normalize_doc_id(item_doc_id)
    if not norm:
        return False
    if norm in allowed_ids:
        return True
    stripped = norm.removesuffix(".jpeg").removesuffix(".jpg").removesuffix(".png")
    return stripped in allowed_ids


def _normalize_candidate(
    raw: Evidence | dict[str, Any],
    query: str,
    scoped: bool,
    rank: int | None = None,
) -> dict[str, Any] | None:
    if isinstance(raw, Evidence):
        chunk_id = raw.chunk_id
        doc_id = raw.doc_id
        text = raw.text
        score = getattr(raw, "score", 0.0)
        metadata = dict(getattr(raw, "metadata", {}) or {})
        pre_rerank_rank = getattr(raw, "pre_rerank_rank", None) or metadata.get("pre_rerank_rank")
        pre_rerank_score = getattr(raw, "pre_rerank_score", None) or metadata.get("pre_rerank_score")
    else:
        chunk_id = raw.get("chunk_id") or raw.get("id")
        doc_id = raw.get("doc_id")
        text = raw.get("text") or raw.get("content") or raw.get("page_content") or ""
        score = raw.get("score", raw.get("hybrid_score", 0.0))
        metadata = dict(raw.get("metadata") or {})
        pre_rerank_rank = raw.get("pre_rerank_rank") or metadata.get("pre_rerank_rank")
        pre_rerank_score = raw.get("pre_rerank_score") or metadata.get("pre_rerank_score")

    if not chunk_id:
        return None

    chunk_id_str = str(chunk_id)
    if not doc_id and ":" in chunk_id_str:
        doc_id = chunk_id_str.split(":", 1)[0]

    try:
        numeric_score = float(score or 0.0)
    except (TypeError, ValueError):
        numeric_score = 0.0

    try:
        numeric_pre_score = (
            float(pre_rerank_score)
            if pre_rerank_score is not None
            else numeric_score
        )
    except (TypeError, ValueError):
        numeric_pre_score = numeric_score

    parent_id = _derive_parent_id(chunk_id_str, {"parent_id": metadata.get("parent_id")})
    effective_ret_rank = pre_rerank_rank if pre_rerank_rank is not None else rank

    return {
        "chunk_id": chunk_id_str,
        "doc_id": _normalize_doc_id(doc_id),
        "parent_id": parent_id,
        "text": text or "",
        "score": numeric_score,
        "pre_rerank_rank": effective_ret_rank,
        "pre_rerank_score": numeric_pre_score,
        "retriever_rank": rank,
        "source_query": query,
        "scoped_search": scoped,
        "metadata": metadata,
    }


def make_rag_strategy_retriever(
    rag_strategy,
    documents: list[Document],
    image: Any = None,
) -> RetrieverFn:
    """Wraps RAGStrategyBase.retrieve(...) for retrieval_node."""

    def _search(query: str, doc_scope: list[str] | None, mode: str) -> list[Evidence]:
        _ = mode
        if doc_scope and len(doc_scope) == 1:
            results = rag_strategy.retrieve(
                query=query,
                documents=documents,
                image=image,
                scope="document",
                doc_id=doc_scope[0],
            ) or []
            allowed = {str(d).strip() for d in doc_scope if str(d).strip()}
            return [item for item in results if _matches_scope(getattr(item, "doc_id", None), allowed)]

        results = rag_strategy.retrieve(
            query=query,
            documents=documents,
            image=image,
            scope="global",
            doc_id=None,
        ) or []
        if doc_scope:
            allowed = {str(d).strip() for d in doc_scope if str(d).strip()}
            return [item for item in results if _matches_scope(getattr(item, "doc_id", None), allowed)]
        return results

    return _search


def make_search_strategy_retriever(
    search_strategy,
    documents: list[Document],
    limit: int = 8,
) -> RetrieverFn:
    """Wraps SearchStrategyBase.search(...) or SearchRerankerPipeline.search(...) for retrieval_node."""
    sig = inspect.signature(search_strategy.search)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    def _call_strategy(query: str, active_scope: list[str] | None) -> list[Any]:
        kwargs: dict[str, Any] = {}

        def _maybe_set(names: list[str], value: Any) -> None:
            for name in names:
                if accepts_var_kw or name in params:
                    kwargs[name] = value
                    return

        _maybe_set(["documents", "docs"], documents)
        _maybe_set(["limit", "top_k", "k"], limit)

        if active_scope and len(active_scope) == 1:
            _maybe_set(["scope", "search_scope"], "document")
            _maybe_set(["doc_id", "document_id"], active_scope[0])
        else:
            _maybe_set(["scope", "search_scope"], "global")

        try:
            return search_strategy.search(query=query, **kwargs)
        except TypeError:
            try:
                return search_strategy.search(query, documents, limit)
            except TypeError:
                return search_strategy.search(query)

    def _search(query: str, doc_scope: list[str] | None, mode: str) -> list[Any]:
        _ = mode
        results = _call_strategy(query, doc_scope) or []
        if doc_scope and not isinstance(results, dict):
            allowed = {str(d).strip() for d in doc_scope if str(d).strip()}
            filtered = []
            for item in results:
                item_doc_id = (
                    item.get("doc_id")
                    if isinstance(item, dict)
                    else getattr(item, "doc_id", None)
                )
                if _matches_scope(item_doc_id, allowed):
                    filtered.append(item)
            return filtered
        return results

    return _search


def _default_search_fn(
    query: str,
    doc_scope: list[str] | None,
    mode: str,
) -> list[Evidence | dict[str, Any]]:
    _ = (query, doc_scope, mode)
    return []


def adapt_to_retriever_fn(
    retriever_or_fn: Any,
    documents: list[Document] | None = None,
    image: Any = None,
    limit: int = 8,
) -> RetrieverFn:
    """Adapt a RAGStrategyBase, SearchStrategyBase, or search callable into RetrieverFn."""
    docs = documents or []
    if retriever_or_fn is None:
        return _default_search_fn

    if hasattr(retriever_or_fn, "retrieve") and callable(retriever_or_fn.retrieve):
        return make_rag_strategy_retriever(retriever_or_fn, docs, image=image)

    if hasattr(retriever_or_fn, "search") and callable(retriever_or_fn.search):
        return make_search_strategy_retriever(retriever_or_fn, docs, limit=limit)

    if callable(retriever_or_fn):
        def _call_raw(query: str, doc_scope: list[str] | None, mode: str) -> list[Any]:
            try:
                return retriever_or_fn(query, doc_scope, mode)
            except TypeError:
                try:
                    return retriever_or_fn(query, doc_scope=doc_scope, documents=docs, limit=limit)
                except TypeError:
                    try:
                        return retriever_or_fn(query, limit=limit)
                    except TypeError:
                        return retriever_or_fn(query)

        return _call_raw

    return _default_search_fn


def make_retrieval_node(search_fn: RetrieverFn = _default_search_fn):
    def retrieval_node(state: AgentRAGState) -> dict[str, Any]:
        queries = state.get("search_queries") or [state["question"]]
        doc_scope = state.get("doc_scope")
        plan = state.get("plan") or {}
        mode = plan.get("mode", "initial_search")
        trace = list(state.get("trace", []))

        merged_by_chunk: dict[str, dict[str, Any]] = {}
        if mode == "retry_search":
            for prev in state.get("candidates") or []:
                cid = prev.get("chunk_id")
                if cid:
                    merged_by_chunk[cid] = dict(prev)

        def _run_pass(active_scope: list[str] | None, scoped_flag: bool) -> int:
            new_or_improved = 0
            for q in queries:
                for idx, raw in enumerate(search_fn(q, active_scope, mode) or [], start=1):
                    norm = _normalize_candidate(raw, query=q, scoped=scoped_flag, rank=idx)
                    if not norm:
                        continue
                    cid = norm["chunk_id"]
                    existing = merged_by_chunk.get(cid)
                    if existing is None or norm["score"] > existing.get("score", 0.0):
                        merged_by_chunk[cid] = norm
                        new_or_improved += 1
            return new_or_improved

        if doc_scope:
            scoped_hits = _run_pass(doc_scope, scoped_flag=True)
            if scoped_hits == 0:
                _run_pass(None, scoped_flag=False)
                trace.append(
                    f"retrieval: scoped={doc_scope} had 0 hits -> fallback global queries={len(queries)}"
                )
            else:
                trace.append(
                    f"retrieval: scoped={doc_scope} queries={len(queries)} unique_candidates={len(merged_by_chunk)}"
                )
        else:
            _run_pass(None, scoped_flag=False)
            trace.append(
                f"retrieval: global queries={len(queries)} unique_candidates={len(merged_by_chunk)}"
            )

        candidates = sorted(
            merged_by_chunk.values(),
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )

        logger.debug(
            "RETRIEVAL_DONE | mode=%s queries=%s merged_candidates=%d top_scores=%s",
            mode,
            queries,
            len(candidates),
            [round(float(c.get("score", 0.0)), 4) for c in candidates[:5]],
        )

        return {"candidates": candidates, "trace": trace}

    return retrieval_node


retrieval_node = make_retrieval_node()