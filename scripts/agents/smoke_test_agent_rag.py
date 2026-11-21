from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import pandas as pd

from src.config import AppConfig
from src.generate.generate_answer import build_model_callable
from src.models.records import Document
from src.agents.graph import build_agent_rag_graph
from src.pipeline.factory import _build_prompt, _build_search
from src.strategies.answers import LLMAnswer

logger = logging.getLogger(__name__)

load_dotenv()

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "application.log"

formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logging.getLogger("src.agents").setLevel(logging.DEBUG)


def load_documents(docs_path: str) -> list[Document]:
    path = Path(docs_path)
    with open(path, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)
    return [
        Document(
            doc_id=doc.get("doc_id"),
            content=doc.get("content", ""),
            qa_pairs=doc.get("qa_pairs", []),
            image_path=doc.get("image_path", None),
        )
        for doc in raw_docs
    ]


def load_labels(labels_path: str) -> list[dict[str, Any]]:
    path = Path(labels_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch smoke test for Agent RAG.")
    parser.add_argument("--input", required=True, help="Path to documents JSON file.")
    parser.add_argument("--labels", required=True, help="Path to evaluation labels JSON file.")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of labeled records to evaluate (default: 25).",
    )
    args = parser.parse_args()

    config = AppConfig.from_env()
    documents = load_documents(args.input)
    labels = load_labels(args.labels)

    target_records = [item for item in labels[: args.limit] if item.get("question")]
    if not target_records:
        raise ValueError(f"No records with a 'question' field found in labels[:{args.limit}].")

    rag_or_search = _build_search(config)
    prompt_strategy = _build_prompt(config)
    model_callable = (
        build_model_callable()
        if getattr(config, "answer_strategy", None) == "llm"
        else None
    )
    answer_strategy = LLMAnswer(model_callable) if model_callable is not None else None

    graph = build_agent_rag_graph(
        config=config,
        rag_or_search=rag_or_search,
        prompt_strategy=prompt_strategy,
        documents=documents,
        model_callable=model_callable,
        answer_strategy=answer_strategy,
    )

    total = len(target_records)
    verified_count = 0
    refused_count = 0

    for idx, record in enumerate(target_records, start=1):
        sample_question = record["question"]
        initial_state = {
            "question": sample_question,
            "attachment": None,
            "plan": {"mode": "initial_search", "focus_gaps": [], "feedback": ""},
            "search_queries": [],
            "doc_scope": None,
            "candidates": [],
            "reranked_docs": [],
            "evidence_sufficient": False,
            "evidence_gaps": [],
            "draft_answer": "",
            "should_refuse": False,
            "refusal_reason": "",
            "citations": [],
            "citation_pairs": [],
            "verified": False,
            "verification_notes": "",
            "attempt": 0,
            "max_attempts": config.agent_max_attempts,
            "trace": [],
            "final_output": {},
        }

        logger.info("RUNNING ITEM %d/%d | question=%s", idx, total, sample_question)
        result = graph.invoke(initial_state)

        for step in result.get("trace", []):
            logger.info("TRACE [%d/%d] | %s", idx, total, step)

        gold_ans = (
            record.get("answer")
            or record.get("gold_answer")
            or record.get("expected_answer")
        )
        logger.info("\n%s", format_question_trace_table(result, gold_answer=gold_ans))

        final_out = result.get("final_output") or {}
        is_verified = bool(
            final_out.get("verified") if isinstance(final_out, dict) else result.get("verified", False)
        )
        is_refused = bool(
            final_out.get("should_refuse") if isinstance(final_out, dict) else result.get("should_refuse", False)
        )
        if is_verified:
            verified_count += 1
        if is_refused:
            refused_count += 1

    logger.info(
        "BATCH SUMMARY ([:%d]) | evaluated=%d verified=%d refused=%d",
        args.limit,
        total,
        verified_count,
        refused_count,
    )


def format_question_trace_table(
    state: dict[str, Any],
    gold_answer: str | None = None,
) -> str:
    """Build a readable per-question table of retrieval ranks, rerank ranks, and answers."""
    candidates = state.get("candidates") or []
    reranked = state.get("reranked_docs") or []
    citations = set(state.get("citations") or [])

    rerank_map: dict[str, tuple[int, float | None]] = {}
    for idx, doc in enumerate(reranked, start=1):
        cid = str(doc.get("chunk_id") or "")
        score = doc.get("rerank_score", doc.get("score"))
        rerank_map[cid] = (idx, float(score) if score is not None else None)

    rows: list[dict[str, Any]] = []
    seen_cids: set[str] = set()

    for idx, cand in enumerate(candidates, start=1):
        cid = str(cand.get("chunk_id") or "")
        seen_cids.add(cid)
        meta = cand.get("metadata") or {}
        r_rank, r_score = rerank_map.get(cid, (None, None))
        ret_rank = (
            cand.get("pre_rerank_rank")
            or meta.get("pre_rerank_rank")
            or cand.get("vector_rank")
            or meta.get("vector_rank")
            or cand.get("bm25_rank")
            or meta.get("bm25_rank")
            or cand.get("retriever_rank")
            or idx
        )
        ret_score = (
            cand.get("pre_rerank_score")
            if cand.get("pre_rerank_score") is not None
            else (
                meta.get("pre_rerank_score")
                if meta.get("pre_rerank_score") is not None
                else cand.get("score", 0.0)
            )
        )
        rows.append(
            {
                "chunk_id": cid,
                "doc_id": cand.get("doc_id"),
                "retriever": cand.get("retriever") or meta.get("retriever") or meta.get("source") or "hybrid",
                "ret_rank": ret_rank,
                "ret_score": round(float(ret_score or 0.0), 4),
                "rerank_rank": r_rank,
                "rerank_score": round(r_score, 4) if r_score is not None else None,
                "cited": cid in citations,
            }
        )

    for idx, doc in enumerate(reranked, start=1):
        cid = str(doc.get("chunk_id") or "")
        if cid not in seen_cids:
            r_score = doc.get("rerank_score", doc.get("score"))
            rows.append(
                {
                    "chunk_id": cid,
                    "doc_id": doc.get("doc_id"),
                    "retriever": "reranked_only",
                    "ret_rank": None,
                    "ret_score": None,
                    "rerank_rank": idx,
                    "rerank_score": round(float(r_score), 4) if r_score is not None else None,
                    "cited": cid in citations,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            by=["rerank_rank", "ret_rank"],
            ascending=[True, True],
            na_position="last",
        )
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 200,
            "display.expand_frame_repr", False,
        ):
            table_str = df.to_string(index=False)
    else:
        table_str = "<no candidates retrieved>"

    final_out = state.get("final_output") or {}
    final_ans = final_out.get("answer") if isinstance(final_out, dict) else state.get("draft_answer", "")
    verified = final_out.get("verified") if isinstance(final_out, dict) else state.get("verified", False)

    lines = [
        "=" * 96,
        f"QUESTION   : {state.get('question', '')}",
        f"VERIFIED   : {verified} | ATTEMPTS: {state.get('attempt', 0)}/{state.get('max_attempts', 2)}",
        f"CITATIONS  : {sorted(citations)}",
        "-" * 96,
        table_str,
        "-" * 96,
        f"FINAL ANS  : {final_ans}",
    ]
    if gold_answer is not None:
        lines.append(f"GOLD ANS   : {gold_answer}")
    lines.append("=" * 96)
    return "\n".join(lines)


if __name__ == "__main__":
    main()