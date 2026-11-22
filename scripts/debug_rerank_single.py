"""Standalone one-question diagnostic for candidate pool & BGEReranker ranking."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import os,inspect

from src.config import AppConfig
from src.data import load_saved_sample
from src.models.records import Document, QAPair
from src.pipeline import build_strategies_from_config
from src.strategies.reranker import BGEReranker, _extract_text

QUERY = "Who is the payee of the check?"
TARGET_CHUNK_ID = "3611.jpeg:child:0:0"
TARGET_DOC_ID = "3611.jpeg"


def _normalize_cid(cid: str | None) -> str:
    if not cid:
        return ""
    return str(cid).replace("🧒", ":child:")


def _load_docs() -> list[Document]:
    sample_500 = Path("data/sample_500/documents.json")
    if sample_500.exists():
        raw = json.loads(sample_500.read_text(encoding="utf-8"))
        return [
            Document(
                doc_id=entry["doc_id"],
                content=entry.get("content", ""),
                qa_pairs=[
                    QAPair(question=q.get("question", ""), answer=q.get("answer", ""))
                    for q in entry.get("qa_pairs", [])
                ],
                image_path=(
                    entry["image_path"].replace("\\", "/")
                    if entry.get("image_path")
                    else None
                ),
            )
            for entry in raw
        ]
    return load_saved_sample()


def _unwrap_base_search(search_obj):
    """Unwrap SearchRerankerPipeline or LimitedSearch if present."""
    curr = search_obj
    seen = set()
    while curr is not None and id(curr) not in seen:
        seen.add(id(curr))
        if hasattr(curr, "base_search"):
            curr = curr.base_search
        elif hasattr(curr, "base"):
            curr = curr.base
        else:
            break
    return curr


def main() -> None:
    config = AppConfig.from_env()
    documents = _load_docs()

    sig = inspect.signature(build_strategies_from_config)
    if "model_callable" in sig.parameters:
        strategies = build_strategies_from_config(
            config,
            model_callable=lambda *args, **kwargs: "",
        )
    else:
        strategies = build_strategies_from_config(config)

    # Locate the underlying search strategy from strategies
    search_strategy = None
    for attr in ("search", "retriever", "searcher"):
        if hasattr(strategies, attr) and getattr(strategies, attr) is not None:
            search_strategy = getattr(strategies, attr)
            break
    if search_strategy is None and hasattr(strategies, "rag"):
        rag = strategies.rag
        for attr in ("search", "retriever", "searcher"):
            if hasattr(rag, attr) and getattr(rag, attr) is not None:
                search_strategy = getattr(rag, attr)
                break
        if search_strategy is None and hasattr(rag, "text"):
            for attr in ("search", "retriever", "searcher"):
                if hasattr(rag.text, attr) and getattr(rag.text, attr) is not None:
                    search_strategy = getattr(rag.text, attr)
                    break

    if search_strategy is None:
        raise RuntimeError(f"Could not locate search strategy on {type(strategies).__name__}")

    base_search = _unwrap_base_search(search_strategy)
    pool_size = getattr(search_strategy, "candidate_pool_size", 50)
    top_k = getattr(search_strategy, "top_k", 5)

    print(f"Query: {QUERY!r}")
    print(f"Target chunk: {TARGET_CHUNK_ID} | Base search: {type(base_search).__name__} | pool={pool_size}, top_k={top_k}")

    # 1. Check raw document text for target doc
    doc_match = next((d for d in documents if d.doc_id == TARGET_DOC_ID), None)
    print(f"Doc {TARGET_DOC_ID} present in loaded docs: {doc_match is not None}")
    if doc_match:
        q_terms = set(re.findall(r"\w+", QUERY.lower()))
        d_terms = set(re.findall(r"\w+", (doc_match.content or "").lower()))
        print(f"Query token overlap with {TARGET_DOC_ID}: {sorted(q_terms & d_terms)}")

    # 2. Fetch candidate pool from base search without reranking
    try:
        candidates = base_search.search(QUERY, documents=documents, limit=pool_size)
    except TypeError:
        candidates = base_search.search(QUERY, limit=pool_size)

    print(f"\nCandidate pool returned: {len(candidates)} items")
    found_idx = None
    for idx, item in enumerate(candidates, start=1):
        cid = _normalize_cid(
            item.get("chunk_id") if isinstance(item, dict) else getattr(item, "chunk_id", None)
        )
        doc_id = item.get("doc_id") if isinstance(item, dict) else getattr(item, "doc_id", None)
        if cid == TARGET_CHUNK_ID or doc_id == TARGET_DOC_ID:
            marker = " <-- TARGET CHUNK" if cid == TARGET_CHUNK_ID else ""
            print(
                f"  pool_idx={idx:2d} | doc={doc_id} | chunk={cid} | "
                f"dense={getattr(item, 'dense_rank', None)} | "
                f"sparse={getattr(item, 'sparse_rank', None)} | "
                f"hybrid={getattr(item, 'hybrid_rank', None)}{marker}"
            )
        if cid == TARGET_CHUNK_ID and found_idx is None:
            found_idx = idx

    if found_idx is None:
        print(f"\n[DIAGNOSIS] {TARGET_CHUNK_ID} was NOT in the top-{pool_size} candidate pool from {type(base_search).__name__}.")
        return

    # 3. Score candidates with BGEReranker (or existing reranker)
    reranker = getattr(search_strategy, "reranker", None)
    if reranker is None or not hasattr(reranker, "model"):
        reranker = BGEReranker()

    pairs = [[QUERY, _extract_text(item)] for item in candidates]
    import torch

    reranker.model.eval()
    with torch.no_grad():
        inputs = reranker.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(reranker.device)
        scores = reranker.model(**inputs).logits.squeeze(-1).float().cpu().numpy()

    ranked = sorted(
        enumerate(zip(candidates, scores), start=1),
        key=lambda x: x[1][1],
        reverse=True,
    )

    print("\nTop 8 after BGEReranker cross-encoder scoring:")
    for rerank_pos, (pool_idx, (item, score)) in enumerate(ranked[:8], start=1):
        cid = _normalize_cid(
            item.get("chunk_id") if isinstance(item, dict) else getattr(item, "chunk_id", None)
        )
        doc_id = item.get("doc_id") if isinstance(item, dict) else getattr(item, "doc_id", None)
        print(
            f"  rerank_#{rerank_pos} (was pool_#{pool_idx:2d}) | score={float(score):8.4f} | "
            f"doc={doc_id} | chunk={cid}"
        )

    for rerank_pos, (pool_idx, (item, score)) in enumerate(ranked, start=1):
        cid = _normalize_cid(
            item.get("chunk_id") if isinstance(item, dict) else getattr(item, "chunk_id", None)
        )
        if cid == TARGET_CHUNK_ID:
            print(
                f"\n[DIAGNOSIS] {TARGET_CHUNK_ID} landed at rerank_#{rerank_pos} "
                f"(pool_#{pool_idx}, score={float(score):.4f}) -> "
                f"{'INSIDE top_k' if rerank_pos <= top_k else f'OUTSIDE top_k={top_k}'}"
            )
            break


if __name__ == "__main__":
    main()