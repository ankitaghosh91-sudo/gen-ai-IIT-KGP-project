import re
from collections.abc import Callable
from typing import Any

from src.agents.state import AgentRAGState

RerankerFn = Callable[[str, list[dict[str, Any]], int], list[dict[str, Any]]]


def _default_rerank_fn(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Fallback reranker sorted by retrieval score when no cross-encoder is passed."""
    _ = query
    ranked = sorted(
        candidates,
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    out: list[dict[str, Any]] = []
    for rank, doc in enumerate(ranked[:top_k], start=1):
        copied = dict(doc)
        copied["rerank_score"] = float(copied.get("score", 0.0))
        copied["rerank_rank"] = rank
        out.append(copied)
    return out


def _diversify_top_k(
    docs: list[dict[str, Any]],
    top_k: int,
    max_per_doc: int = 2,
) -> list[dict[str, Any]]:
    """Keep top-ranked items while capping chunks from the same document when alternatives exist."""
    if top_k <= 0 or not docs:
        return []

    selected: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    doc_counts: dict[str, int] = {}

    for doc in docs:
        doc_key = str(doc.get("doc_id") or doc.get("parent_id") or doc.get("chunk_id") or id(doc))
        count = doc_counts.get(doc_key, 0)
        if count < max_per_doc and len(selected) < top_k:
            selected.append(doc)
            doc_counts[doc_key] = count + 1
        else:
            overflow.append(doc)

    for doc in overflow:
        if len(selected) >= top_k:
            break
        selected.append(doc)

    return selected


def _tokens(text: str) -> set[str]:
    return {
        tok.lower()
        for tok in re.findall(r"[A-Za-z0-9_]{3,}", text or "")
    }


def _assess_coverage(
    question: str,
    reranked_docs: list[dict[str, Any]],
    plan: dict[str, Any],
) -> tuple[bool, list[str]]:
    gaps: list[str] = []

    non_empty_docs = [d for d in reranked_docs if (d.get("text") or "").strip()]
    if not non_empty_docs:
        gaps.append("No candidate chunks with non-empty text were retrieved or reranked.")
        return False, gaps

    q_tokens = _tokens(question)
    combined_text = " ".join(d.get("text", "") for d in non_empty_docs[:5])
    ctx_tokens = _tokens(combined_text)

    overlap_ratio = (len(q_tokens & ctx_tokens) / len(q_tokens)) if q_tokens else 1.0

    entities = plan.get("entities") or {}
    required_terms = [
        *(entities.get("quoted_terms") or []),
        *(entities.get("doc_refs") or []),
    ]
    missing_terms = [
        term
        for term in required_terms
        if term.lower() not in combined_text.lower()
    ]
    if missing_terms:
        gaps.append(f"Missing explicit reference/entity terms in top evidence: {missing_terms}")

    top_doc = non_empty_docs[0]
    top_score = float(top_doc.get("rerank_score", top_doc.get("score", 0.0)))

    if len(q_tokens) >= 3 and overlap_ratio < 0.12 and not missing_terms:
        gaps.append(
            f"Low term overlap ({overlap_ratio:.2f}) between question and top reranked chunks."
        )

    if top_score < -3.5 or (top_score < -2.0 and overlap_ratio < 0.20):
        gaps.append(f"Weak top reranker score ({top_score:.4f}) for query relevance.")

    return len(gaps) == 0, gaps


def make_evidence_analysis_node(
    rerank_fn: RerankerFn = _default_rerank_fn,
    top_k: int = 5,
    max_per_doc: int = 2,
):
    def evidence_analysis_node(state: AgentRAGState) -> dict[str, Any]:
        question = state["question"]
        queries = state.get("search_queries") or [question]
        primary_query = queries[0]
        candidates = state.get("candidates") or []
        plan = state.get("plan") or {}
        trace = list(state.get("trace", []))

        reranked = rerank_fn(primary_query, candidates, top_k * 2) or []
        diversified = _diversify_top_k(reranked, top_k=top_k, max_per_doc=max_per_doc)

        normalized_reranked: list[dict[str, Any]] = []
        for idx, doc in enumerate(diversified, start=1):
            item = dict(doc)
            if "rerank_score" not in item:
                item["rerank_score"] = float(item.get("score", 0.0))
            item["rerank_rank"] = idx
            normalized_reranked.append(item)

        sufficient, gaps = _assess_coverage(question, normalized_reranked, plan)
        trace.append(
            f"evidence_analysis: reranked={len(normalized_reranked)} sufficient={sufficient} gaps={len(gaps)}"
        )

        return {
            "reranked_docs": normalized_reranked,
            "evidence_sufficient": sufficient,
            "evidence_gaps": gaps,
            "trace": trace,
        }

    return evidence_analysis_node


evidence_analysis_node = make_evidence_analysis_node()