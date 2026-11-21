from typing import Any, Literal, TypedDict


class TaskPlan(TypedDict):
    mode: Literal["initial_search", "retry_search", "finalize"]
    focus_gaps: list[str]
    feedback: str


class AgentRAGState(TypedDict):
    question: str
    attachment: str | None
    plan: TaskPlan
    search_queries: list[str]
    doc_scope: list[str] | None
    candidates: list[dict[str, Any]]
    reranked_docs: list[dict[str, Any]]
    evidence_sufficient: bool
    evidence_gaps: list[str]
    draft_answer: str
    should_refuse: bool
    refusal_reason: str
    citations: list[str]
    citation_pairs: list[dict[str, str]]
    verified: bool
    verification_notes: str
    attempt: int
    max_attempts: int
    trace: list[str]
    final_output: dict[str, Any]