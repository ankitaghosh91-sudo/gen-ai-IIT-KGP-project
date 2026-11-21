"""Build/verify finance_sample_500_parent_child and check child chunk sizes."""

from __future__ import annotations

import json
from pathlib import Path
import chromadb

from src.config import AppConfig
from src.models.records import Document
from src.pipeline.factory import _build_chunker, _build_embedder
from src.strategies.search import ChromaSearch  # adjust import if ChromaSearch lives elsewhere

DOCS_PATH = Path("data/sample_500/documents.json")
LIMIT_DOCS = None  # set to None to index all documents in data/sample_500/documents.json
MAX_CHILD_LEN = 350


def load_documents(path: Path, limit: int | None = None) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        docs = raw.get("documents") or raw.get("data") or raw.get("items") or []
    else:
        docs = raw
    return docs[:limit] if limit is not None else docs


def extract_doc_id_and_text(doc: dict | str, fallback_idx: int) -> tuple[str, str]:
    if isinstance(doc, str):
        return f"doc_{fallback_idx}", doc
    doc_id = str(
        doc.get("doc_id")
        or doc.get("id")
        or doc.get("document_id")
        or doc.get("file_name")
        or f"doc_{fallback_idx}"
    )
    text = (
        doc.get("text")
        or doc.get("content")
        or doc.get("body")
        or doc.get("ocr_text")
        or ""
    )
    return doc_id, str(text)


def main() -> None:
    config = AppConfig.from_env()
    documents = load_documents(DOCS_PATH, limit=LIMIT_DOCS)
    print(f"Loaded {len(documents)} documents from {DOCS_PATH}")

    chunker = _build_chunker(config)
    embedder = _build_embedder(config)

    child_lengths: list[int] = []
    over_limit: list[tuple[int, str, str]] = []
    sample_ids: list[str] = []

    for idx, doc in enumerate(documents):
        doc_id, text = extract_doc_id_and_text(doc, idx)
        rows = chunker.split_with_hierarchy(text, doc_id)
        for row in rows:
            cid = row["child_id"]
            ctext = row["child_text"] or ""
            clen = len(ctext)
            child_lengths.append(clen)
            if len(sample_ids) < 10:
                sample_ids.append(cid)
            if clen > MAX_CHILD_LEN:
                over_limit.append((clen, cid, ctext[:80]))

    if child_lengths:
        print(
            f"Child chunks ({len(child_lengths)}) -> "
            f"min={min(child_lengths)}, avg={sum(child_lengths)/len(child_lengths):.1f}, "
            f"max={max(child_lengths)} chars"
        )
        print(f"Child chunks > {MAX_CHILD_LEN} chars: {len(over_limit)}")
        for clen, cid, preview in over_limit[:5]:
            print(f"  > {MAX_CHILD_LEN}: id={cid} len={clen} preview={preview!r}")
    else:
        print("No child chunks produced—check extract_doc_id_and_text keys against documents.json.")
    print("Sample child IDs:", sample_ids)


    search = ChromaSearch(
        path=config.chroma_path,
        collection_name=config.chroma_collection,
        embedder=embedder,
        chunker=chunker,
    )
    doc_objects = [
        Document(
            doc_id=str(
                doc.get("doc_id") or doc.get("id") or doc.get("file_name") or f"doc_{i}"
            )
            if isinstance(doc, dict)
            else getattr(doc, "doc_id", f"doc_{i}"),
            content=str(
                doc.get("content") or doc.get("text") or doc.get("body") or ""
            )
            if isinstance(doc, dict)
            else getattr(doc, "content", ""),
            qa_pairs=(doc.get("qa_pairs", []) if isinstance(doc, dict) else getattr(doc, "qa_pairs", [])),
            image_path=(doc.get("image_path") if isinstance(doc, dict) else getattr(doc, "image_path", None)),
        )
        for i, doc in enumerate(documents)
    ]
    search.index_documents(doc_objects)

    client = chromadb.PersistentClient(path=config.chroma_path)
    col = client.get_collection(config.chroma_collection)
    print(f"Collection '{col.name}' now has {col.count()} stored items.")
    print("\n========== INDEXING COMPLETED ==========")
    print("Documents indexed:", len(documents))
    print("Stored chunks:", search.collection.count())
    print("Collection:", search.collection.name)
    print("Chroma path:", config.chroma_path)




if __name__ == "__main__":
    main()