from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import logging
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


def _safe_setattr(target: Any, attr: str, value: Any) -> None:
    """Set an attribute on mutable objects or instance __dict__ without triggering broken setters."""
    if attr == "rank_gain":
        return
    target_dict = getattr(target, "__dict__", None)
    if isinstance(target_dict, dict):
        target_dict[attr] = value
        return
    if is_dataclass(target):
        return
    try:
        setattr(target, attr, value)
    except AttributeError:
        pass


def _stamp_pre_rerank(items: list[Any]) -> None:
    """Record 1-based pre-rerank rank and original score before reranking reorders items."""
    for idx, item in enumerate(items, start=1):
        orig_score = (
            item.get("score")
            if isinstance(item, dict)
            else getattr(item, "score", None)
        )
        if isinstance(item, dict):
            item.setdefault("pre_rerank_rank", idx)
            if orig_score is not None:
                item.setdefault("pre_rerank_score", orig_score)
            meta = item.setdefault("metadata", {})
            if isinstance(meta, dict):
                meta.setdefault("pre_rerank_rank", idx)
                if orig_score is not None:
                    meta.setdefault("pre_rerank_score", orig_score)
        else:
            if getattr(item, "pre_rerank_rank", None) is None:
                _safe_setattr(item, "pre_rerank_rank", idx)
            if orig_score is not None and getattr(item, "pre_rerank_score", None) is None:
                _safe_setattr(item, "pre_rerank_score", orig_score)
            meta = getattr(item, "metadata", None)
            if isinstance(meta, dict):
                meta.setdefault("pre_rerank_rank", idx)
                if orig_score is not None:
                    meta.setdefault("pre_rerank_score", orig_score)


def _extract_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("text") or item.get("content") or item.get("page_content") or "")
    return str(getattr(item, "text", None) or getattr(item, "content", "") or "")


class BGEReranker:
    def __init__(self, model_name="BAAI/bge-reranker-base"):
        print(f"Loading local reranker model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        # Use GPU if available for fast scoring
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def rerank(self, query: str, results: list, top_k: int = 5) -> list:
        """Reranks retrieval results using BGE cross-encoder."""
        if not results:
            return []

        _stamp_pre_rerank(results)

        # Prepare pairs of [query, document_text]
        pairs = [[query, _extract_text(item)] for item in results]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            # Cross-encoder outputs a single relevance score per pair
            scores = self.model(**inputs).logits.squeeze(-1).float().cpu().numpy()

        # Attach rerank scores back to the result items
        scored_results = list(zip(results, scores))

        # Sort by score descending
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Update item scores and rerank ranks, then slice to final top_k
        reranked_items = []
        for rerank_idx, (item, score) in enumerate(scored_results[:top_k], start=1):
            pre_rank = (
                item.get("pre_rerank_rank")
                if isinstance(item, dict)
                else getattr(item, "pre_rerank_rank", None)
            )
            rank_gain = (
                (int(pre_rank) - rerank_idx)
                if pre_rank is not None
                else None
            )

            if isinstance(item, dict):
                updated = dict(item)
                updated["score"] = float(score)
                updated["rerank_rank"] = rerank_idx
                updated["rank_gain"] = rank_gain
            elif is_dataclass(item):
                updated = type(item).__new__(type(item))
                item_dict = getattr(item, "__dict__", None)
                if isinstance(item_dict, dict):
                    for k, v in item_dict.items():
                        if k != "rank_gain":
                            updated.__dict__[k] = copy.deepcopy(v)
                else:
                    for f in fields(item):
                        if f.name != "rank_gain" and hasattr(item, f.name):
                            object.__setattr__(updated, f.name, getattr(item, f.name))

                _safe_setattr(updated, "score", float(score))
                _safe_setattr(updated, "rerank_rank", rerank_idx)
            else:
                updated = copy.copy(item)
                updated.score = float(score)
                _safe_setattr(updated, "rerank_rank", rerank_idx)

            reranked_items.append(updated)

        return reranked_items


class SearchRerankerPipeline:
    """Fetch a wider candidate pool from a base retriever and rerank top-k items."""

    def __init__(
        self,
        base_search: Any,
        reranker: Any,
        top_k: int = 5,
        candidate_pool_size: int = 50,
    ) -> None:
        self.base_search = base_search
        self.reranker = reranker
        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size

    def _fetch_candidates(self, query: str, documents: list[Any] | None = None) -> list[Any]:
        """Call base_search across common parameter conventions safely."""
        pool_k = self.candidate_pool_size

        # Try top_k with documents
        try:
            return self.base_search.search(query, documents=documents, top_k=pool_k)
        except TypeError:
            pass

        # Try limit with documents
        try:
            return self.base_search.search(query, documents=documents, limit=pool_k)
        except TypeError:
            pass

        # Try top_k without documents
        try:
            return self.base_search.search(query, top_k=pool_k)
        except TypeError:
            pass

        # Try limit without documents
        try:
            return self.base_search.search(query, limit=pool_k)
        except TypeError:
            pass

        # Fallback to positional/default search
        if documents is not None:
            try:
                return self.base_search.search(query, documents)
            except TypeError:
                pass

        return self.base_search.search(query)

    def search(
        self,
        query: str,
        documents: list[Any] | None = None,
        top_k: int | None = None,
        limit: int | None = None,
    ) -> list[Any]:
        effective_top_k = (
            top_k
            if top_k is not None
            else (limit if limit is not None else self.top_k)
        )
        candidates = self._fetch_candidates(query=query, documents=documents)
        if candidates:
            _stamp_pre_rerank(candidates)
        logger.info(
            "Reranking %d candidates (pool_size=%d) to top_k=%d",
            len(candidates),
            self.candidate_pool_size,
            effective_top_k,
        )
        if not candidates:
            return []
        return self.reranker.rerank(query=query, results=candidates, top_k=effective_top_k)