"""RAG routing strategies."""

import inspect
import re
from typing import Any, Callable

from src.models import Document, Evidence
from src.strategies.base import RAGStrategyBase
from src.strategies.search import ChromaSearch


def _call_search(
    search: Any,
    query: str,
    documents: list[Document],
    scope: str = "global",
    doc_id: str | None = None,
    limit: int | None = None,
) -> list[Evidence]:
    sig = inspect.signature(search.search)
    params = sig.parameters
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    kwargs: dict[str, Any] = {}
    if accepts_kwargs or "documents" in params:
        kwargs["documents"] = documents
    if accepts_kwargs or "scope" in params:
        kwargs["scope"] = scope
    if accepts_kwargs or "doc_id" in params:
        kwargs["doc_id"] = doc_id
    if limit is not None and (accepts_kwargs or "limit" in params):
        kwargs["limit"] = limit

    if accepts_kwargs or "query" in params:
        results = search.search(query=query, **kwargs)
    else:
        results = search.search(query, **kwargs)

    # If the search pipeline does not handle document scoping itself, filter here.
    if scope == "document" and doc_id and "scope" not in params and not accepts_kwargs:
        results = [
            item for item in results if getattr(item, "doc_id", None) == doc_id
        ]

    return results


class SimpleRAG(RAGStrategyBase):
    def __init__(self, search=None):
        self.search = search or ChromaSearch()

    def retrieve(
        self,
        query: str,
        documents: list[Document],
        image: Any = None,
        scope: str = "global",
        doc_id: str | None = None,
    ) -> list[Evidence]:
        return _call_search(
            self.search,
            query=query,
            documents=documents,
            scope=scope,
            doc_id=doc_id,
        )


class VisualRAG(RAGStrategyBase):
    def __init__(self, vision: Callable | None = None):
        self.vision = vision

    def retrieve(
        self, query: str, documents: list[Document], image: Any = None
    ) -> list[Evidence]:
        if self.vision is None or image is None:
            return []
        return self.vision(query, image)


class CombinedRAG(RAGStrategyBase):
    def __init__(self, text=None, visual=None):
        self.text = text or SimpleRAG()
        self.visual = visual or VisualRAG()

    def retrieve(
        self,
        query: str,
        documents: list[Document],
        image: Any = None,
        scope: str = "global",
        doc_id: str | None = None,
    ) -> list[Evidence]:
        return (
            self.text.retrieve(
                query,
                documents,
                scope=scope,
                doc_id=doc_id,
            )
            + self.visual.retrieve(query, documents, image)
        )


class AutoRAG(CombinedRAG):
    def retrieve(
        self,
        query: str,
        documents: list[Document],
        image: Any = None,
        scope: str = "global",
        doc_id: str | None = None,
    ) -> list[Evidence]:
        if image is None:
            return self.text.retrieve(
                query,
                documents,
                scope=scope,
                doc_id=doc_id,
            )

        return super().retrieve(
            query,
            documents,
            image,
            scope=scope,
            doc_id=doc_id,
        )


class MultiStepRAG(SimpleRAG):
    def retrieve(
        self,
        query: str,
        documents: list[Document],
        image: Any = None,
        scope: str = "global",
        doc_id: str | None = None,
    ) -> list[Evidence]:
        clauses = [
            part.strip()
            for part in re.split(r"\band\b|[,?]", query, flags=re.I)
            if part.strip()
        ]

        found = []
        seen = set()

        for clause in [query, *clauses]:
            for item in _call_search(
                self.search,
                query=clause,
                documents=documents,
                scope=scope,
                doc_id=doc_id,
            ):
                key = (item.doc_id, item.question)

                if key not in seen:
                    found.append(item)
                    seen.add(key)

        return found


class AgenticRAG(SimpleRAG):
    def retrieve(
        self,
        query: str,
        documents: list[Document],
        image: Any = None,
        scope: str = "global",
        doc_id: str | None = None,
    ) -> list[Evidence]:
        found = _call_search(
            self.search,
            query=query,
            documents=documents,
            limit=8,
            scope=scope,
            doc_id=doc_id,
        )

        words = set(re.findall(r"\w+", query.lower()))

        found.sort(
            key=lambda item: len(
                words
                & set(
                    re.findall(
                        r"\w+",
                        f"{item.question} {item.text}".lower(),
                    )
                )
            ),
            reverse=True,
        )

        return found[:4]