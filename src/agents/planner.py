import re
from typing import Any, Literal

from src.agents.state import AgentRAGState

QueryIntent = Literal[
    "entity_lookup",
    "financial_field",
    "analytical_inference",
    "general_qa",
]

_STOP_REF_WORDS = {
    "invoice",
    "invoices",
    "document",
    "documents",
    "check",
    "checks",
    "po",
    "order",
    "orders",
    "possible",
    "issuer",
    "amount",
    "total",
    "payment",
    "terms",
    "purpose",
    "address",
    "recipient",
    "payee",
}

_RETRY_GAP_STOPWORDS = {
    "missing",
    "explicit",
    "reference",
    "entity",
    "terms",
    "top",
    "evidence",
    "weak",
    "reranker",
    "score",
    "query",
    "relevance",
    "upstream",
    "marked",
    "insufficient",
    "gap",
    "gaps",
    "unresolved",
    "low",
    "term",
    "overlap",
    "between",
    "question",
    "and",
    "chunks",
    "for",
    "the",
    "or",
    *_STOP_REF_WORDS,
}

_QUESTION_STOPWORDS = {
    "what",
    "who",
    "whom",
    "whose",
    "which",
    "where",
    "when",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "by",
    "with",
    "as",
    "that",
    "this",
    "it",
    "per",
}


def _detect_intent(question: str) -> QueryIntent:
    q_lower = question.lower().strip()
    if any(
        token in q_lower
        for token in (
            "infer",
            "why is it important",
            "significance",
            "implication",
            "compare",
            "increase",
        )
    ):
        return "analytical_inference"
    if any(
        token in q_lower
        for token in (
            "who is the issuer",
            "issuer",
            "payee",
            "recipient",
            "address",
            "vendor",
            "individual",
        )
    ):
        return "entity_lookup"
    if any(
        token in q_lower
        for token in (
            "amount",
            "payment term",
            "purpose of the payment",
            "total",
            "gross",
            "check",
            "invoice",
            "purchase order",
        )
    ):
        return "financial_field"
    return "general_qa"


def _extract_entities(question: str) -> dict[str, list[str]]:
    quoted = [
        q.strip()
        for q in re.findall(r"['\"]([^'\"]+)['\"]", question)
        if q.strip()
    ]

    # Require an explicit separator/ID/number marker so words like "invoice"
    # or "possible" are not truncated into fake document IDs.
    raw_refs = re.findall(
        r"\b(?:doc|invoice|inv|po|check)\s*(?:#|no\.?|num(?:ber)?|id)?\s*[:#-]\s*([A-Za-z0-9_-]+)\b",
        question,
        flags=re.IGNORECASE,
    )
    doc_refs = [
        ref
        for ref in raw_refs
        if ref.lower() not in _STOP_REF_WORDS and (re.search(r"\d", ref) or len(ref) >= 3)
    ]

    numbers = re.findall(r"\b\d{3,}\b", question)
    return {
        "quoted_terms": list(dict.fromkeys(quoted)),
        "doc_refs": list(dict.fromkeys(doc_refs)),
        "numeric_tokens": list(dict.fromkeys(numbers)),
    }


def _infer_doc_scope(
    state: AgentRAGState,
    entities: dict[str, list[str]],
) -> list[str] | None:
    existing_scope = state.get("doc_scope")
    if existing_scope:
        cleaned = [
            str(s).strip()
            for s in existing_scope
            if s and str(s).strip().lower() not in _STOP_REF_WORDS
        ]
        if cleaned:
            return list(dict.fromkeys(cleaned))

    if entities["doc_refs"]:
        return list(dict.fromkeys(entities["doc_refs"]))

    return None


def _distinctive_question_terms(question: str) -> list[str]:
    terms: list[str] = []
    for tok in re.findall(r"[A-Za-z0-9_-]{3,}", question or ""):
        low = tok.lower()
        if low in _QUESTION_STOPWORDS or low in _STOP_REF_WORDS:
            continue
        if low not in {t.lower() for t in terms}:
            terms.append(tok)
    return terms


def _build_search_queries(
    question: str,
    intent: QueryIntent,
    entities: dict[str, list[str]],
    plan: dict[str, Any],
) -> list[str]:
    clean_question = question.strip()
    queries: list[str] = [clean_question]

    anchor_terms: list[str] = []
    anchor_terms.extend(entities["quoted_terms"])
    anchor_terms.extend(entities["doc_refs"])

    if intent == "entity_lookup":
        anchor_terms.extend(["issuer", "issued by", "organization", "address"])
    elif intent == "financial_field":
        anchor_terms.extend(["amount", "payment terms", "purpose", "total"])
    elif intent == "analytical_inference":
        anchor_terms.extend(["compliance", "verification", "terms", "summary"])

    keyword_query = " ".join(dict.fromkeys([clean_question, *anchor_terms])).strip()
    if keyword_query and keyword_query not in queries:
        queries.append(keyword_query)

    if plan.get("mode") == "retry_search":
        raw_gap_text = " ".join(str(g) for g in (plan.get("focus_gaps") or []))
        gap_tokens = [
            tok
            for tok in re.findall(r"[A-Za-z_-]{3,}", raw_gap_text)
            if tok.lower() not in _RETRY_GAP_STOPWORDS
        ]
        q_terms = _distinctive_question_terms(clean_question)
        retry_query = " ".join(
            dict.fromkeys([clean_question, *anchor_terms, *q_terms, *gap_tokens[:4]])
        ).strip()
        if retry_query and retry_query not in queries:
            queries.append(retry_query)

    return queries[:3]


def query_planning_node(state: AgentRAGState) -> dict[str, Any]:
    """Understands intent, extracts entities, forms search queries, and sets doc scope."""
    question = state["question"]
    plan = dict(state.get("plan") or {})
    trace = list(state.get("trace", []))

    intent = _detect_intent(question)
    entities = _extract_entities(question)
    doc_scope = _infer_doc_scope(state, entities)
    search_queries = _build_search_queries(question, intent, entities, plan)

    updated_plan = {
        **plan,
        "intent": intent,
        "entities": entities,
    }
    trace.append(
        f"query_planning: intent={intent} queries={len(search_queries)} doc_scope={doc_scope}"
    )

    return {
        "plan": updated_plan,
        "search_queries": search_queries,
        "doc_scope": doc_scope,
        "trace": trace,
    }