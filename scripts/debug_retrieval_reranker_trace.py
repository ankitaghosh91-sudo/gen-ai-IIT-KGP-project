from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
import math
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import pandas as pd

from src.config import AppConfig
from src.models.records import Document
from src.pipeline.factory import _build_search

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _clean_json_value(value: Any) -> Any:
    """Convert pandas/float NaNs to None so JSON output is strictly valid (null)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _clean_json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json_value(v) for v in value]
    return value


def _df_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    raw_records = df.to_dict(orient="records")
    return [{k: _clean_json_value(v) for k, v in row.items()} for row in raw_records]


def _format_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    return df.to_string(index=False, na_rep="-")


def load_documents(docs_path: str) -> list[Document]:
    path = Path(docs_path)
    if not path.exists():
        raise FileNotFoundError(f"Documents file not found: {docs_path}")
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
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _call_search(strategy: Any, query: str, documents: list[Any], limit: int) -> list[Any]:
    for kwargs in (
        {"query": query, "documents": documents, "limit": limit},
        {"query": query, "documents": documents, "top_k": limit},
        {"query": query, "top_k": limit},
    ):
        try:
            return strategy.search(**kwargs)
        except TypeError:
            continue
    return strategy.search(query)


def _index_if_supported(strategy: Any, documents: list[Document]) -> None:
    target = getattr(strategy, "base", strategy)
    if type(target).__name__ == "BM25Search":
        return
    if hasattr(strategy, "index_documents"):
        try:
            strategy.index_documents(documents)
        except NotImplementedError:
            return


def _normalize_hits(results: list[Any]) -> list[dict[str, Any]]:
    """Extract (rank, chunk_id, doc_id, score, rerank_score) uniformly from dicts or objects."""
    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(results, start=1):
        if isinstance(item, dict):
            cid = item.get("chunk_id")
            did = item.get("doc_id")
            score = item.get("score")
            r_score = item.get("rerank_score")
        else:
            cid = getattr(item, "chunk_id", None)
            did = getattr(item, "doc_id", None)
            score = getattr(item, "score", None)
            r_score = getattr(item, "rerank_score", None)

        if isinstance(cid, (list, tuple, set)):
            cid = next((str(x).strip() for x in cid if x is not None and str(x).strip()), None)
        elif cid is not None:
            cid = str(cid).strip()

        if not cid:
            continue

        if not did and ":" in cid:
            did = cid.split(":", 1)[0]

        clean_score = _clean_json_value(score)
        clean_r_score = _clean_json_value(r_score)

        rows.append(
            {
                "rank": rank,
                "chunk_id": cid,
                "doc_id": did or "",
                "score": round(float(clean_score), 4) if clean_score is not None else None,
                "rerank_score": round(float(clean_r_score), 4) if clean_r_score is not None else None,
            }
        )
    return rows


def _extract_gold_ids(record: dict[str, Any]) -> list[str]:
    rel = record.get("relevant_chunk_id") or record.get("relevant_chunk_ids") or record.get("chunk_ids") or []
    partial = record.get("partially_relevant_chunk_id") or []
    if isinstance(rel, str):
        rel = [rel]
    if isinstance(partial, str):
        partial = [partial]
    combined = [str(c).strip() for c in list(rel) + list(partial) if str(c).strip()]
    return list(dict.fromkeys(combined))


def _build_strategy_table(
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    hybrid_hits: list[dict[str, Any]],
    gold_ids: set[str],
    max_rows: int = 30,
) -> pd.DataFrame:
    """Table 1: Side-by-side rank comparison across Dense, Sparse, and Hybrid search."""
    dense_map = {h["chunk_id"]: h for h in dense_hits}
    sparse_map = {h["chunk_id"]: h for h in sparse_hits}
    hybrid_map = {h["chunk_id"]: h for h in hybrid_hits}

    ordered_ids: list[str] = []
    for source in (hybrid_hits, dense_hits, sparse_hits):
        for h in source:
            if h["chunk_id"] not in ordered_ids:
                ordered_ids.append(h["chunk_id"])

    table_rows: list[dict[str, Any]] = []
    for cid in ordered_ids[:max_rows]:
        d = dense_map.get(cid)
        s = sparse_map.get(cid)
        hy = hybrid_map.get(cid)
        table_rows.append(
            {
                "chunk_id": cid,
                "gold": cid in gold_ids,
                "dense_rank": int(d["rank"]) if d else None,
                "sparse_rank": int(s["rank"]) if s else None,
                "hybrid_rank": int(hy["rank"]) if hy else None,
            }
        )
    return pd.DataFrame(table_rows)


def _build_reranker_table(
    reranker_input_hits: list[dict[str, Any]],
    reranker_output_hits: list[dict[str, Any]],
    gold_ids: set[str],
    top_display: int = 10,
) -> pd.DataFrame:
    """Table 2: Top-K reranked candidates with pool rank, hybrid rank, and rank shift."""
    in_map = {h["chunk_id"]: h for h in reranker_input_hits}
    top_outputs = reranker_output_hits[:top_display]

    table_rows: list[dict[str, Any]] = []
    for out in top_outputs:
        cid = out["chunk_id"]
        inp = in_map.get(cid)
        in_rank = int(inp["rank"]) if inp else None
        out_rank = int(out["rank"])
        rank_shift = (in_rank - out_rank) if in_rank is not None else None
        r_score = out["rerank_score"] if out["rerank_score"] is not None else out["score"]
        table_rows.append(
            {
                "chunk_id": cid,
                "gold": cid in gold_ids,
                "in_pool_rank": in_rank,
                "hybrid_rank": in_rank,
                "rerank_rank": out_rank,
                "rerank_score": r_score,
                "rank_shift": rank_shift,
            }
        )
    return pd.DataFrame(table_rows)


def _format_item_block(
    idx: int,
    total: int,
    record: dict[str, Any],
    question: str,
    gold_ids: set[str],
    df_strategies: pd.DataFrame,
    df_reranker: pd.DataFrame,
    pool_len: int,
    rerank_display_len: int,
) -> str:
    strategies_str = _format_table(df_strategies)
    reranker_str = _format_table(df_reranker)
    return (
        "====================================================================================================\n"
        f"ITEM {idx}/{total} | eval_id={record.get('eval_id')} | gold_doc={record.get('doc_id')} | gold_chunks={sorted(gold_ids)}\n"
        f"QUESTION: {question}\n"
        f"GOLD ANS: {record.get('gold_answer')}\n"
        "----------------------------------------------------------------------------------------------------\n"
        f"[TABLE 1] RETRIEVAL RANKS (Dense vs Sparse vs Hybrid | rows={len(df_strategies)})\n"
        f"{strategies_str}\n"
        "----------------------------------------------------------------------------------------------------\n"
        f"[TABLE 2] RERANKER TOP-{rerank_display_len} (from pool n={pool_len})\n"
        f"{reranker_str}\n"
        "===================================================================================================="
    )


def trace_retrieval_and_reranker(
    docs_path: str,
    labels_path: str,
    output_path: str,
    text_output_path: str | None = None,
    limit: int | None = 5,
    table_rows: int = 30,
    reranker_top_display: int = 10,
    index_docs: bool = False,
) -> dict[str, Any]:
    documents = load_documents(docs_path)
    labels = load_labels(labels_path)
    if limit is not None:
        labels = labels[:limit]

    base_config = AppConfig().from_env()
    dense_strategy = _build_search(replace(base_config, search_strategy="chroma"))
    sparse_strategy = _build_search(replace(base_config, search_strategy="bm25"))
    hybrid_strategy = _build_search(replace(base_config, search_strategy="hybrid"))
    reranker_strategy = _build_search(replace(base_config, search_strategy="hybrid_reranker"))

    if index_docs:
        logger.info("Indexing %d documents into stores before tracing...", len(documents))
        _index_if_supported(dense_strategy, documents)

    reranker_pool_size = (
        getattr(reranker_strategy, "candidate_pool_size", None)
        or getattr(reranker_strategy, "pool_size", None)
        or getattr(base_config, "reranker_pool_size", 40)
    )
    reranker_top_k = getattr(reranker_strategy, "top_k", None) or base_config.retrieval_limit

    logger.info(
        "Initialized strategies | retrieval_limit=%s | reranker_pool_size=%s | reranker_top_k=%s | reranker_top_display=%s",
        base_config.retrieval_limit,
        reranker_pool_size,
        reranker_top_k,
        reranker_top_display,
    )

    trace_records: list[dict[str, Any]] = []
    text_blocks: list[str] = [
        f"Retrieval & Reranker Trace Report\n"
        f"docs_path={docs_path} | labels_path={labels_path} | items={len(labels)}\n"
        f"retrieval_limit={base_config.retrieval_limit} | reranker_pool_size={reranker_pool_size} | reranker_top_display={reranker_top_display}\n"
    ]

    for idx, record in enumerate(labels, start=1):
        question = record.get("question", "")
        if not question:
            continue

        gold_ids = set(_extract_gold_ids(record))

        dense_hits = _normalize_hits(
            _call_search(dense_strategy, question, documents, limit=base_config.retrieval_limit)
        )
        sparse_hits = _normalize_hits(
            _call_search(sparse_strategy, question, documents, limit=base_config.retrieval_limit)
        )

        max_hybrid_limit = max(int(base_config.retrieval_limit), int(reranker_pool_size))
        all_hybrid_hits = _normalize_hits(
            _call_search(hybrid_strategy, question, documents, limit=max_hybrid_limit)
        )
        hybrid_hits = all_hybrid_hits[: int(base_config.retrieval_limit)]
        reranker_input_hits = all_hybrid_hits[: int(reranker_pool_size)]

        reranker_output_hits = _normalize_hits(
            _call_search(reranker_strategy, question, documents, limit=int(reranker_top_k))
        )

        df_strategies = _build_strategy_table(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            hybrid_hits=hybrid_hits,
            gold_ids=gold_ids,
            max_rows=table_rows,
        )
        df_reranker = _build_reranker_table(
            reranker_input_hits=reranker_input_hits,
            reranker_output_hits=reranker_output_hits,
            gold_ids=gold_ids,
            top_display=reranker_top_display,
        )

        item_block = _format_item_block(
            idx=idx,
            total=len(labels),
            record=record,
            question=question,
            gold_ids=gold_ids,
            df_strategies=df_strategies,
            df_reranker=df_reranker,
            pool_len=len(reranker_input_hits),
            rerank_display_len=len(df_reranker),
        )
        logger.info("\n%s", item_block)
        text_blocks.append(item_block)

        trace_records.append(
            {
                "item": idx,
                "eval_id": record.get("eval_id"),
                "doc_id": record.get("doc_id"),
                "question_id": record.get("question_id"),
                "question": question,
                "gold_answer": record.get("gold_answer"),
                "gold_chunk_ids": sorted(gold_ids),
                "table_1_strategies": _df_to_json_records(df_strategies),
                "table_2_reranker": _df_to_json_records(df_reranker),
                "raw": {
                    "dense": dense_hits,
                    "sparse": sparse_hits,
                    "hybrid": hybrid_hits,
                    "reranker_input": reranker_input_hits,
                    "reranker_output": reranker_output_hits,
                },
            }
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _clean_json_value(
        {
            "limit": limit,
            "retrieval_limit": base_config.retrieval_limit,
            "reranker_pool_size": reranker_pool_size,
            "reranker_top_k": reranker_top_k,
            "reranker_top_display": reranker_top_display,
            "items": trace_records,
        }
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)

    txt_path = Path(text_output_path) if text_output_path else out_path.with_suffix(".txt")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(text_blocks) + "\n")

    logger.info("Saved retrieval + reranker JSON report to %s", out_path)
    logger.info("Saved structured text trace report to %s", txt_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace Dense, Sparse, Hybrid, and Reranker ranks & scores."
    )
    parser.add_argument("--input", required=True, help="Path to documents JSON file.")
    parser.add_argument("--labels", required=True, help="Path to evaluation gold set JSON file.")
    parser.add_argument("--output", required=True, help="Path to save JSON trace report.")
    parser.add_argument(
        "--text-output",
        default=None,
        help="Optional path to save structured .txt table report (defaults to <output>.txt).",
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of gold items to run (0 for all).")
    parser.add_argument("--table-rows", type=int, default=30, help="Max rows printed in Table 1.")
    parser.add_argument(
        "--reranker-top-display",
        type=int,
        default=10,
        help="Number of top reranked chunks shown in Table 2 / .txt report.",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index documents into ChromaDB and ParentChunkStore before running trace.",
    )
    args = parser.parse_args()

    limit = None if args.limit == 0 else args.limit
    trace_retrieval_and_reranker(
        docs_path=args.input,
        labels_path=args.labels,
        output_path=args.output,
        text_output_path=args.text_output,
        limit=limit,
        table_rows=args.table_rows,
        reranker_top_display=args.reranker_top_display,
        index_docs=args.index,
    )


if __name__ == "__main__":
    main()