"""One small injectable function that runs the copilot from input to response."""

from dataclasses import dataclass, field
from typing import Any, Callable


from src.models import Document, Evidence
from src.strategies import (
    AnswerStrategyBase,
    AutoRAG,
    ConversationMemory,
    DatasetAnswer,
    EvaluatorBase,
    MemoryStrategyBase,
    PromptStrategyBase,
    RAGStrategyBase,
    SimpleEvaluator,
    SimplePrompt,
)

import logging
from time import perf_counter

logger = logging.getLogger(__name__)

@dataclass
class PipelineStrategies:
    """Swap one component at a time; the rest of the flow remains unchanged."""

    rag: RAGStrategyBase = field(default_factory=AutoRAG)
    answerer: AnswerStrategyBase = field(default_factory=DatasetAnswer)
    prompt: PromptStrategyBase = field(default_factory=SimplePrompt)
    memory: MemoryStrategyBase = field(default_factory=ConversationMemory)
    evaluator: EvaluatorBase = field(default_factory=SimpleEvaluator)
    request_validator: Callable[[str], None] | None = None
    response_validator: Callable[[dict, list[Evidence]], dict] | None = None
    router: Callable[[str, Any], str] | None = None


def validate_request(question: str) -> None:
    if not question or not question.strip():
        raise ValueError("Enter a question.")


def validate_response(answer: dict, evidence: list[Evidence]) -> dict:
    if not isinstance(answer, dict):
        raise ValueError("Answer strategy must return a dictionary.")
    if not isinstance(answer.get("answer"), str):
        raise ValueError("Answer strategy must return an answer string.")
    if not isinstance(answer.get("should_refuse"), bool):
        raise ValueError("Answer strategy must return a should_refuse boolean.")
    if not isinstance(answer.get("reason", ""), str):
        raise ValueError("Answer strategy refusal reason must be text.")
    valid_ids = {item.doc_id for item in evidence}
    citations = answer.get("citations", [])
    if not isinstance(citations, list) or any(
        not isinstance(item, dict) or item.get("doc_id") not in valid_ids for item in citations
    ):
        return {
            "answer": "I cannot verify the answer citation.",
            "citations": [],
            "should_refuse": True,
            "reason": "Citation validation failed.",
        }
    return answer


def route_request(question: str, image: Any = None) -> str:
    return "visual" if image is not None else "text"


def run_pipeline(
    question: str,
    documents: list[Document],
    session_id: str = "default",
    expected_answer: str | None = None,
    image: Any = None,
    strategies: PipelineStrategies | None = None,
    scope: str = "global",
    doc_id: str | None = None,
) -> dict:
    """Run validate → route → memory → retrieve → answer → evaluate → validate."""
    choices = strategies or PipelineStrategies()
    validate = choices.request_validator or validate_request
    response_check = choices.response_validator or validate_response
    router = choices.router or route_request

    validate(question)
    logger.info(
        "PIPELINE_STARTED | session_id=%s | scope=%s | doc_id=%s",
        session_id,
        scope,
        doc_id,
    )
    route = router(question, image)
    history = choices.memory.get(session_id)
    evidence = choices.rag.retrieve(
        question,
        documents,
        image,
        scope=scope,
        doc_id=doc_id,
    )
    logger.info(
        "RETRIEVAL_COMPLETED | count=%d | chunks=%s",
        len(evidence),
        [
            {
                "doc_id": item.doc_id,
                "chunk_id": item.chunk_id,
                "score": round(item.score, 4),
            }
            for item in evidence
        ],
    )
    logger.info(
        "ANSWER_GENERATION_STARTED | strategy=%s | evidence_count=%d",
        type(choices.answerer).__name__,
        len(evidence),
    )

    answer_started = perf_counter()

    try:
        answer = choices.answerer.answer(
            question=question,
            evidence=evidence,
            prompt=choices.prompt,
            history=history,
            documents=documents,
        )
    except Exception:
        logger.exception("ANSWER_GENERATION_FAILED")
        raise
    
    logger.info(
        "ANSWER_GENERATION_COMPLETED | duration_ms=%.2f | refused=%s | citations=%s",
        (perf_counter() - answer_started) * 1000,
        answer.get("should_refuse"),
        answer.get("citations"),
    )

    metrics = choices.evaluator.evaluate(answer, evidence, expected_answer)
    answer = response_check(answer, evidence)
    choices.memory.add(session_id, question, answer["answer"])

    return {
        "answer": answer,
        "route": route,
        "evidence": evidence,
        "history": history,
        "metrics": metrics,
        "steps": [
            "validate request",
            f"route: {route}",
            "load memory",
            "search documents",
            "answer with prompt strategy",
            "evaluate",
            "validate response",
        ],
    }
