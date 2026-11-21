"""Adapter between LLMClient and the answer strategy."""

import json
import logging
import re
import time

logger = logging.getLogger(__name__)


def _extract_json_object(raw_text: str) -> str:
    """Strip markdown code fences or surrounding prose to isolate a JSON object."""
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


def _attempt_truncated_json_repair(payload: str) -> dict | None:
    """Best-effort repair when output was cut off before closing quotes/braces."""
    s = payload.strip()
    if not s.startswith("{"):
        return None

    # Close an unclosed string literal if odd number of unescaped quotes
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

    # Remove trailing comma if present before closing braces/brackets
    s = re.sub(r",\s*$", "", s)

    # Balance square brackets and curly braces
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


class LLMModelAdapter:
    """Convert a text prompt into a structured, grounded LLM response."""

    SYSTEM_PROMPT = """
You are a financial document question-answering assistant.

Answer the user's question using ONLY the retrieved evidence.

Rules:
1. Do not use outside knowledge or make unsupported assumptions.
2. If the evidence does not sufficiently answer the question,
   refuse to answer.
3. Cite the exact doc_id and chunk_id of the evidence supporting
   your answer.
4. Never invent document IDs or chunk IDs.
5. Keep the answer concise, direct, and factual (3-5 sentences max).
6. Ensure all strings are properly escaped and JSON is complete.

Return ONLY a valid JSON object with this exact structure:

{
    "answer": "Your answer, or an empty string if refusing.",
    "citations": [
        {
            "doc_id": "document ID",
            "chunk_id": "chunk ID"
        }
    ],
    "should_refuse": false,
    "reason": ""
}

For insufficient evidence:
- should_refuse must be true.
- citations must be an empty list.
- reason must explain why the evidence is insufficient.
"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def _parse_and_validate(self, content: str) -> dict:
        json_payload = _extract_json_object(content)

        try:
            result = json.loads(json_payload)
        except json.JSONDecodeError as exc:
            repaired = _attempt_truncated_json_repair(json_payload)
            if repaired is not None:
                logger.warning("LLM_JSON_REPAIRED_TRUNCATED_OUTPUT")
                result = repaired
            else:
                logger.exception(
                    "LLM_JSON_PARSE_FAILED | raw_preview=%r",
                    content[:400],
                )
                raise ValueError(
                    "LLM returned an invalid JSON response."
                ) from exc

        if not isinstance(result, dict):
            raise ValueError("LLM response must be a JSON object.")

        required_fields = {
            "answer": str,
            "citations": list,
            "should_refuse": bool,
            "reason": str,
        }

        for field, expected_type in required_fields.items():
            if field not in result:
                raise ValueError(f"LLM response is missing field: {field}")
            if not isinstance(result[field], expected_type):
                raise ValueError(f"Invalid type for LLM response field: {field}")

        for citation in result["citations"]:
            if not isinstance(citation, dict):
                raise ValueError("Each citation must be an object.")
            for field in ("doc_id", "chunk_id"):
                if not isinstance(citation.get(field), str) or not citation[field].strip():
                    raise ValueError(f"Each citation must contain a valid {field}.")

        if result["should_refuse"]:
            if result["citations"]:
                raise ValueError("A refused answer must not contain citations.")
        else:
            if not result["answer"].strip():
                raise ValueError("LLM returned an empty answer.")
            if not result["citations"]:
                raise ValueError("A non-refused answer must contain citations.")

        logger.info(
            "LLM_JSON_PARSED | refused=%s | citation_count=%d",
            result["should_refuse"],
            len(result["citations"]),
        )
        return result

    def __call__(self, prompt: str) -> dict:
        logger.info("LLM_PROVIDER_CALL_STARTED")
        started = time.perf_counter()

        content = self.llm_client.generate(
            messages=[
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.0,
            max_tokens=1200,
        )
        logger.info(
            "LLM_PROVIDER_CALL_COMPLETED | duration_ms=%.2f | response_chars=%d",
            (time.perf_counter() - started) * 1000,
            len(content),
        )

        return self._parse_and_validate(content)