"""Chunking strategies."""

import re

from src.strategies.base import ChunkingStrategyBase
from langchain_text_splitters import RecursiveCharacterTextSplitter


class WholeDocumentChunking(ChunkingStrategyBase):
    def split(self, text: str) -> list[str]:
        return [text] if text else []


class FixedSizeChunking(ChunkingStrategyBase):
    def __init__(self, size: int = 800, overlap: int = 100):
        if size < 1 or overlap < 0 or overlap >= size:
            raise ValueError("Require size > overlap >= 0")
        self.size = size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        step = self.size - self.overlap
        return [
            text[index : index + self.size]
            for index in range(0, len(text), step)
            if text[index : index + self.size].strip()
            and (index == 0 or len(text[index : index + self.size]) > self.overlap)
        ]


class ParagraphChunking(ChunkingStrategyBase):
    def split(self, text: str) -> list[str]:
        return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


class RecursiveChunking(ChunkingStrategyBase):

    def __init__(self, chunk_size: int = 500, overlap: int = 100):

        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0
        ):
            raise ValueError("chunk_size must be a positive integer")

        if (
            isinstance(overlap, bool)
            or not isinstance(overlap, int)
            or overlap < 0
            or overlap >= chunk_size
        ):
            raise ValueError("overlap must be non-negative and less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def split(self, text: str) -> list[str]:

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if not text.strip():
            return []

        return self.splitter.split_text(text)


class ParentChildChunker(ChunkingStrategyBase):
    """Splits text into large parent chunks and smaller child chunks for hierarchical retrieval."""

    def __init__(
        self,
        parent_size: int = 1200,
        parent_overlap: int = 150,
        child_size: int = 300,
        child_overlap: int = 50,
    ):
        if parent_size <= child_size:
            raise ValueError("parent_size must be greater than child_size")

        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def split_with_hierarchy(self, text: str, doc_id: str) -> list[dict]:
        """Splits a document text into hierarchical parent-child structures."""
        if not isinstance(text, str) or not text.strip():
            return []

        parents = self.parent_splitter.split_text(text)
        structured_chunks = []

        for p_idx, parent_text in enumerate(parents):
            parent_id = f"{doc_id}:parent:{p_idx}"
            children = self.child_splitter.split_text(parent_text)

            for c_idx, child_text in enumerate(children):
                child_id = f"{doc_id}:child:{p_idx}:{c_idx}"
                structured_chunks.append(
                    {
                        "child_id": child_id,
                        "child_text": child_text,
                        "parent_id": parent_id,
                        "parent_text": parent_text,
                        "chunk_index": c_idx,
                    }
                )

        return structured_chunks

    def split(self, text: str) -> list[str]:
        """Fallback to child splitting to maintain ChunkingStrategyBase compatibility."""
        return self.child_splitter.split_text(text)