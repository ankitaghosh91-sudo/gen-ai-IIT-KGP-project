"""Answer-generation strategies."""

from collections import Counter
from typing import Callable

from src.models.records import Evidence , Document
from src.strategies.base import AnswerStrategyBase
from src.strategies.helpers import meaningful_words, refusal


import logging
import time

logger = logging.getLogger(__name__)

class LLMAnswer(AnswerStrategyBase):
    def __init__(self, model: Callable[[str], dict]):
        self.model = model

    def answer(
        self,
        question: str,
        evidence: list[Evidence],
        prompt,
        history=None,
        documents: list[Document] | None = None,
    ) -> dict:

        logger.info(
            "LLM_ANSWER_STARTED | evidence_count=%d | chunk_ids=%s",
            len(evidence),
            [item.chunk_id for item in evidence],
        )

        if not evidence:
            logger.warning(
                "LLM_ANSWER_REFUSED | reason=no_retrieved_evidence"
            )
            return refusal("No document evidence was found.")

        context = "\n\n".join(
            f"Document ID: {item.doc_id}\n"
            f"Chunk ID: {item.chunk_id}\n"
            f"Evidence:\n{item.text}"
            for item in evidence
        )

        request = prompt.build(question, evidence)

        if history:
            request = f"Conversation history: {history}\n{request}"

        logger.info(
            "LLM_MODEL_CALL_STARTED | context_chunks=%d",
            len(evidence),
        )

        started = time.perf_counter()

        try:
            result = self.model(
                request + "\nRetrieved context:\n" + context
            )
            # Validate that every citation belongs to retrieved evidence.
            retrieved_pairs = {
                (item.doc_id, item.chunk_id)
                for item in evidence
                if item.chunk_id is not None
            }

            citations = result.get("citations", [])

            for citation in citations:
                cited_pair = (
                    citation.get("doc_id"),
                    citation.get("chunk_id"),
                )

                if cited_pair not in retrieved_pairs:
                    logger.error(
                        "INVALID_LLM_CITATION | cited=%s | retrieved=%s",
                        cited_pair,
                        retrieved_pairs,
                    )
                    raise ValueError(
                        f"LLM cited a chunk that was not retrieved: {cited_pair}"
                    )

            logger.info(
                "LLM_CITATIONS_VALIDATED | citation_count=%d",
                len(citations),
            )
        except Exception:
            logger.exception("LLM_MODEL_CALL_FAILED")
            raise

        logger.info(
            "LLM_MODEL_CALL_COMPLETED | duration_ms=%.2f | refused=%s",
            (time.perf_counter() - started) * 1000,
            result.get("should_refuse"),
        )

        return result


class DatasetAnswer(AnswerStrategyBase):
    def answer(
        self,
        question: str,
        evidence: list[Evidence],
        prompt,
        history=None,
        documents: list[Document] | None = None,
    ) -> dict:
        if not evidence:
            return refusal("No document evidence was found.")

        if not documents:
            return refusal("Original dataset QA pairs are unavailable.")

        prompt.build(question, evidence)

        question_terms = meaningful_words(question)
        evidence_doc_ids = {item.doc_id for item in evidence}

        best_match = None
        best_overlap = 0

        for document in documents:
            # Only consider QA pairs from documents represented in retrieved evidence.
            if document.doc_id not in evidence_doc_ids:
                continue

            for qa_pair in document.qa_pairs:
                overlap = len(
                    question_terms & meaningful_words(qa_pair.question)
                )

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = (document.doc_id, qa_pair.answer)

        if best_match is None:
            return refusal(
                "The retrieved documents do not contain a matching dataset answer."
            )

        doc_id, answer = best_match

        if not answer:
            return refusal("The matching dataset answer is empty.")

        return {
            "answer": answer,
            "citations": [{"doc_id": doc_id}],
            "should_refuse": False,
            "reason": "",
        }


class CriticAnswer(AnswerStrategyBase):
    def __init__(self, answerer: AnswerStrategyBase | None = None):
        self.answerer = answerer or DatasetAnswer()

    def answer(
        self,
        question: str,
        evidence: list[Evidence],
        prompt,
        history=None,
        documents: list[Document] | None = None,
    ) -> dict:
        result = self.answerer.answer(
            question=question,
            evidence=evidence,
            prompt=prompt,
            history=history,
            documents=documents,
        )

        valid_ids = {item.doc_id for item in evidence}

        if any(
            citation.get("doc_id") not in valid_ids
            for citation in result["citations"]
        ):
            return refusal("Answer citation did not match retrieved evidence.")

        return result



class VotingAnswer(AnswerStrategyBase):
    def __init__(self, answerers: list[AnswerStrategyBase]):
        if not answerers:
            raise ValueError("Provide at least one answer strategy.")
        self.answerers = answerers

    def answer(
        self,
        question: str,
        evidence: list[Evidence],
        prompt,
        history=None,
        documents: list[Document] | None = None,
    ) -> dict:
        answers = [
            item.answer(
                question=question,
                evidence=evidence,
                prompt=prompt,
                history=history,
                documents=documents,
            )
            for item in self.answerers
        ]

        counts = Counter(item["answer"] for item in answers)
        winning_answer, votes = counts.most_common(1)[0]

        if votes <= len(self.answerers) // 2:
            return refusal("Answer strategies did not agree.")

        return next(
            item for item in answers if item["answer"] == winning_answer
        )
