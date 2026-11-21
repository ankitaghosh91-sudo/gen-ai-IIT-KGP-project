import re
from collections.abc import Callable
from typing import Any, Literal

from src.agents.state import AgentRAGState

VerifyFn = Callable[[AgentRAGState], tuple[bool, str, list[str], list[str]]]


def _tokens(text: str) -> set[str]:
    return {tok.lower() for tok in re.findall(r"[A-Za-z0-9_]{3,}", text or "")}


def _default_verify_fn(
    state: AgentRAGState,
) -> tuple[bool, str, list[str], list[str]]:
    draft = (state.get("draft_answer") or "").strip()
    should_refuse = bool(state.get("should_refuse", False))
    refusal_reason = (state.get("refusal_reason") or "").strip()
    reranked_docs = state.get("reranked_docs") or []
    raw_citations = state.get("citations") or []
    evidence_sufficient = bool(state.get("evidence_sufficient", False))
    existing_gaps = list(state.get("evidence_gaps") or [])

    valid_chunk_ids = {
        str(d.get("chunk_id"))
        for d in reranked_docs
        if d.get("chunk_id")
    }
    valid_citations = [
        str(cid) for cid in raw_citations if str(cid) in valid_chunk_ids
    ]
    invalid_citations = [
        str(cid) for cid in raw_citations if str(cid) not in valid_chunk_ids
    ]

    issues: list[str] = []

    if should_refuse:
        issues.append(
            f"Answer refused due to insufficient evidence: {refusal_reason or 'unspecified'}"
        )
    elif not draft:
        issues.append("Draft answer is empty.")

    if not should_refuse and not valid_citations:
        issues.append("No valid citations matching reranked chunk IDs.")

    if invalid_citations:
        issues.append(f"Dropped invalid citations: {invalid_citations}")

    evidence_text = " ".join(
        (d.get("text") or "") for d in reranked_docs if (d.get("text") or "").strip()
    )
    ans_tokens = _tokens(draft)
    ev_tokens = _tokens(evidence_text)
    if not should_refuse and ans_tokens and ev_tokens:
        overlap = len(ans_tokens & ev_tokens) / max(1, len(ans_tokens))
        if overlap < 0.20:
            issues.append(
                f"Low factual grounding overlap ({overlap:.2f}) between draft answer and evidence."
            )

    if not evidence_sufficient:
        issues.append("Upstream evidence analysis marked evidence as insufficient.")

    updated_gaps = list(dict.fromkeys([*existing_gaps, *issues]))
    verified = (not should_refuse) and len(issues) == 0
    notes = (
        f"Verified with {len(valid_citations)} valid citation(s)."
        if verified
        else "; ".join(issues)
    )
    return verified, notes, valid_citations, updated_gaps


def make_verification_node(verify_fn: VerifyFn = _default_verify_fn):
    def verification_node(state: AgentRAGState) -> dict[str, Any]:
        trace = list(state.get("trace", []))
        verified, notes, valid_citations, updated_gaps = verify_fn(state)
        trace.append(
            f"verification: verified={verified} valid_citations={len(valid_citations)} gaps={len(updated_gaps)}"
        )
        return {
            "verified": verified,
            "verification_notes": notes,
            "citations": valid_citations,
            "evidence_gaps": updated_gaps,
            "trace": trace,
        }

    return verification_node


def route_after_verification(
    state: AgentRAGState,
) -> Literal["supervisor", "finalize"]:
    if state.get("verified", False):
        return "finalize"
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 2)
    if attempt < max_attempts:
        return "supervisor"
    return "finalize"


verification_node = make_verification_node()