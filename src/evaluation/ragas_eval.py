"""RAG-grounded evaluation module for RAG pipeline outputs using LLMClient."""

import json
import logging
import re
from typing import Any

from src.evaluation.llm_client import LLMClient

logger = logging.getLogger(__name__)


def _extract_chunk_texts(
    evidence_list: list[Any],
    max_chunks: int = 5,
    max_chars_per_chunk: int = 900,
) -> list[str]:
    texts: list[str] = []
    for item in (evidence_list or [])[:max_chunks]:
        if isinstance(item, dict):
            val = item.get("text") or item.get("content") or item.get("page_content") or ""
        else:
            val = (
                getattr(item, "text", None)
                or getattr(item, "content", None)
                or getattr(item, "page_content", None)
                or ""
            )
        cleaned = str(val).strip()
        if cleaned:
            texts.append(cleaned[:max_chars_per_chunk])
    return texts


def _extract_json_object(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        return ""

    fenced = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    fenced = re.sub(r"\n?```\s*$", "", fenced).strip()

    if fenced.startswith("{") and fenced.endswith("}"):
        return fenced

    start = fenced.find("{")
    if start == -1:
        return fenced

    end = fenced.rfind("}")
    if end != -1 and end > start:
        return fenced[start : end + 1].strip()

    return fenced[start:].strip()


def _attempt_json_repair(payload: str) -> dict[str, Any] | None:
    s = payload.strip()
    if not s.startswith("{"):
        return None

    in_string = False
    escaped = False
    for ch in s:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string

    if in_string:
        s += '"'

    s = re.sub(r",\s*$", "", s)
    open_brackets = s.count("[") - s.count("]")
    if open_brackets > 0:
        s += "]" * open_brackets

    s = re.sub(r",\s*([}\]])", r"\1", s)
    open_braces = s.count("{") - s.count("}")
    if open_braces > 0:
        s += "}" * open_braces

    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _parse_judge_scores(raw_text: str) -> dict[str, Any]:
    json_payload = _extract_json_object(raw_text)
    try:
        parsed = json.loads(json_payload)
    except json.JSONDecodeError as exc:
        repaired = _attempt_json_repair(json_payload)
        if repaired is not None:
            logger.warning("RAGAS_JUDGE_JSON_REPAIRED")
            return repaired
        raise ValueError(f"Judge returned invalid JSON: {raw_text[:200]!r}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Judge output must be a JSON object.")
    return parsed


def _clamp_score(value: Any, default: float = 0.0) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if 1.0 < num <= 10.0:
        num = num / 10.0
    return round(max(0.0, min(1.0, num)), 4)


def run_ragas_evaluation(
    records: list[dict[str, Any]],
    *,
    client: LLMClient | None = None,
    max_chunks: int = 5,
    max_chars_per_chunk: int = 900,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """
    Evaluate captured RAG turns on faithfulness, answer_relevancy, and context_precision.

    Each record in `records` can provide:
      - question: str
      - answer: str
      - evidence: list[Any] (or `contexts: list[str]`)
      - should_refuse: bool (optional)
      - reason: str (optional)
    """
    if not records:
        return {}, []

    judge = client or LLMClient()
    per_row: list[dict[str, Any]] = []

    faithfulness_scores: list[float] = []
    relevancy_scores: list[float] = []
    context_precision_scores: list[float] = []

    for idx, rec in enumerate(records, start=1):
        question = str(rec.get("question") or "").strip()
        answer = str(rec.get("answer") or "").strip()
        should_refuse = bool(rec.get("should_refuse", False))
        reason = str(rec.get("reason") or "").strip()

        contexts = (
            rec.get("contexts")
            if isinstance(rec.get("contexts"), list)
            else _extract_chunk_texts(
                rec.get("evidence") or [],
                max_chunks=max_chunks,
                max_chars_per_chunk=max_chars_per_chunk,
            )
        )
        joined_context = "\n---\n".join(contexts) if contexts else "(No retrieved context)"
        effective_answer = answer or (f"[Refused: {reason}]" if should_refuse else "")

        prompt = (
            "You are evaluating a financial document RAG system.\n"
            "Score each metric on a scale from 0.0 to 1.0:\n"
            '1. "faithfulness": Are all factual claims in the answer supported by the retrieved context '
            "(or is a refusal properly justified when context lacks the answer)?\n"
            '2. "answer_relevancy": Does the response directly address the user\'s question?\n'
            '3. "context_precision": How relevant and signal-rich are the retrieved context chunks '
            "for answering the question?\n\n"
            "Return ONLY a valid JSON object with keys:\n"
            '{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "notes": "brief reason"}\n\n'
            f"Question: {question}\n"
            f"Response: {effective_answer}\n"
            f"Retrieved Context:\n{joined_context}\n"
        )

        try:
            raw_resp = judge.generate(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an objective RAG evaluation judge. Respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=500,
            )
            parsed = _parse_judge_scores(raw_resp)
            faith = _clamp_score(parsed.get("faithfulness", 0.0))
            rel = _clamp_score(parsed.get("answer_relevancy", 0.0))
            ctx_prec = _clamp_score(parsed.get("context_precision", 0.0))
            notes = str(parsed.get("notes") or "").strip()
        except Exception as exc:
            logger.warning("RAGAS_ROW_EVAL_FAILED | row=%d | error=%s", idx, exc)
            faith, rel, ctx_prec = 0.0, 0.0, 0.0
            notes = f"Evaluation error: {exc}"

        faithfulness_scores.append(faith)
        relevancy_scores.append(rel)
        context_precision_scores.append(ctx_prec)

        per_row.append(
            {
                "query_num": idx,
                "question": question,
                "faithfulness": faith,
                "answer_relevancy": rel,
                "context_precision": ctx_prec,
                "notes": notes,
            }
        )

    count = len(records)
    summary = {
        "faithfulness": round(sum(faithfulness_scores) / count, 4),
        "answer_relevancy": round(sum(relevancy_scores) / count, 4),
        "context_precision": round(sum(context_precision_scores) / count, 4),
    }
    return summary, per_row