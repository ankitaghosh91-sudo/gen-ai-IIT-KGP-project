from collections.abc import Callable
from typing import Any

from src.agents.state import AgentRAGState
from src.models.records import Document, Evidence
from src.strategies.base import AnswerStrategyBase, PromptStrategyBase

GenerateFn = Callable[[AgentRAGState], dict[str, Any]]


def _docs_to_evidence(docs: list[dict[str, Any]], max_docs: int) -> list[Evidence]:
    items: list[Evidence] = []
    for doc in docs[:max_docs]:
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        items.append(
            Evidence(
                doc_id=str(doc.get("doc_id") or ""),
                chunk_id=str(doc.get("chunk_id") or ""),
                text=text,
            )
        )
    return items


def make_answer_strategy_generator(
    answer_strategy: AnswerStrategyBase,
    prompt_strategy: PromptStrategyBase,
    documents: list[Document] | None = None,
    max_docs: int = 5,
) -> GenerateFn:
    """Adapts AnswerStrategyBase.answer(...) to AgentRAGState."""

    def _generate(state: AgentRAGState) -> dict[str, Any]:
        reranked_docs = [
            d for d in (state.get("reranked_docs") or []) if (d.get("text") or "").strip()
        ][:max_docs]
        evidence_items = _docs_to_evidence(reranked_docs, max_docs=max_docs)

        result = answer_strategy.answer(
            question=state["question"],
            evidence=evidence_items,
            prompt=prompt_strategy,
            documents=documents,
        )

        raw_citations = result.get("citations") or []
        citation_pairs = [
            {
                "doc_id": str(c.get("doc_id", "")),
                "chunk_id": str(c.get("chunk_id", "")),
            }
            for c in raw_citations
            if isinstance(c, dict)
        ]
        chunk_ids: list[str] = []
        for pair in citation_pairs:
            cid = pair["chunk_id"]
            if cid and cid not in chunk_ids:
                chunk_ids.append(cid)

        should_refuse = bool(result.get("should_refuse", False))
        reason = str(result.get("reason") or "")
        answer_text = str(result.get("answer") or "").strip()

        if should_refuse and not answer_text:
            answer_text = f"Insufficient retrieved evidence: {reason}".strip()

        return {
            "draft_answer": answer_text,
            "should_refuse": should_refuse,
            "refusal_reason": reason,
            "citations": chunk_ids,
            "citation_pairs": citation_pairs,
        }

    return _generate


def _default_generate_fn(state: AgentRAGState) -> dict[str, Any]:
    docs = [
        d for d in (state.get("reranked_docs") or []) if (d.get("text") or "").strip()
    ][:5]
    if not docs:
        return {
            "draft_answer": "",
            "should_refuse": True,
            "refusal_reason": "No relevant document chunks were retrieved.",
            "citations": [],
            "citation_pairs": [],
        }
    pairs = [
        {
            "doc_id": str(d.get("doc_id") or ""),
            "chunk_id": str(d.get("chunk_id") or ""),
        }
        for d in docs
        if d.get("chunk_id")
    ]
    cids = list(dict.fromkeys(p["chunk_id"] for p in pairs if p["chunk_id"]))
    summary = "\n".join(f"- ({d.get('chunk_id')}) {(d.get('text') or '').strip()}" for d in docs[:3])
    return {
        "draft_answer": f"Based on retrieved evidence:\n{summary}",
        "should_refuse": False,
        "refusal_reason": "",
        "citations": cids,
        "citation_pairs": pairs,
    }


def make_answer_generation_node(generate_fn: GenerateFn = _default_generate_fn):
    def answer_generation_node(state: AgentRAGState) -> dict[str, Any]:
        trace = list(state.get("trace", []))
        out = generate_fn(state)
        trace.append(
            f"answer_generation: refused={out.get('should_refuse', False)} citations={len(out.get('citations', []))}"
        )
        return {**out, "trace": trace}

    return answer_generation_node


answer_generation_node = make_answer_generation_node()