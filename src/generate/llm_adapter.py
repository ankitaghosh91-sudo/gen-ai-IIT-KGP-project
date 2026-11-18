
"""Adapter between LLMClient and the answer strategy."""

import json
import logging
import time

logger = logging.getLogger(__name__)

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
5. Keep the answer concise and factual.

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
            max_tokens=500,
        )
        logger.info(
            "LLM_PROVIDER_CALL_COMPLETED | duration_ms=%.2f | response_chars=%d",
            (time.perf_counter() - started) * 1000,
            len(content),
        )

        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.exception("LLM_JSON_PARSE_FAILED")
            raise ValueError(
                "LLM returned an invalid JSON response."
            ) from exc

        if not isinstance(result, dict):
            raise ValueError("LLM response must be a JSON object.")

        logger.info(
            "LLM_JSON_PARSED | refused=%s | citation_count=%d",
            result["should_refuse"],
            len(result["citations"]),
        )

        required_fields = {
            "answer": str,
            "citations": list,
            "should_refuse": bool,
            "reason": str,
        }

        for field, expected_type in required_fields.items():
            if field not in result:
                raise ValueError(
                    f"LLM response is missing field: {field}"
                )

            if not isinstance(result[field], expected_type):
                raise ValueError(
                    f"Invalid type for LLM response field: {field}"
                )

        for citation in result["citations"]:
            if not isinstance(citation, dict):
                raise ValueError("Each citation must be an object.")

            for field in ("doc_id", "chunk_id"):
                if not isinstance(citation.get(field), str) or not citation[field].strip():
                    raise ValueError(
                        f"Each citation must contain a valid {field}."
                    )

        if result["should_refuse"]:
            if result["citations"]:
                raise ValueError(
                    "A refused answer must not contain citations."
                )
        else:
            if not result["answer"].strip():
                raise ValueError("LLM returned an empty answer.")

            if not result["citations"]:
                raise ValueError(
                    "A non-refused answer must contain citations."
                )

        return result
