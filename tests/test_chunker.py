
import pytest

from src.models.records import Document
from src.strategies.chunking import RecursiveChunking


def test_invalid_parameters():
    # Invalid chunk size
    with pytest.raises(ValueError):
        RecursiveChunking(chunk_size=0)

    with pytest.raises(ValueError):
        RecursiveChunking(chunk_size=-100)

    # Invalid overlap
    with pytest.raises(ValueError):
        RecursiveChunking(chunk_size=500, overlap=-10)

    with pytest.raises(ValueError):
        RecursiveChunking(chunk_size=500, overlap=500)

    with pytest.raises(ValueError):
        RecursiveChunking(chunk_size=500, overlap=600)


def test_empty_or_whitespace_input():
    chunker = RecursiveChunking()

    assert chunker.split("") == []
    assert chunker.split("   ") == []
    assert chunker.split("\n\t  ") == []


def test_text_length_edge_cases():
    # Text shorter than chunk size
    chunker = RecursiveChunking(chunk_size=50, overlap=10)

    short_text = "Hello world"
    assert chunker.split(short_text) == [short_text]

    # Text exactly equal to chunk size
    chunker = RecursiveChunking(chunk_size=100, overlap=20)

    exact_text = "A" * 100
    assert chunker.split(exact_text) == [exact_text]


def test_multichunk_and_overlap():
    chunker = RecursiveChunking(chunk_size=10, overlap=3)

    text = "ABCDEFGHIJKLMNOPQRST"
    chunks = chunker.split(text)

    assert len(chunks) > 1

    # All chunks should contain meaningful text
    assert all(chunk.strip() for chunk in chunks)

    # Verify that adjacent chunks have actual overlap.
    # Recursive splitting may produce less than the requested overlap.
    overlap_found = any(
        chunks[i][-3:] == chunks[i + 1][:3]
        for i in range(len(chunks) - 1)
    )

    assert overlap_found


def test_invalid_text_type():
    chunker = RecursiveChunking()

    with pytest.raises(TypeError):
        chunker.split(None)

    with pytest.raises(TypeError):
        chunker.split(12345)


def test_zero_overlap():
    chunker = RecursiveChunking(chunk_size=10, overlap=0)

    text = "ABCDEFGHIJKLMNOPQRST"
    chunks = chunker.split(text)

    assert len(chunks) > 1
    assert "".join(chunks) == text


def test_chunk_document():
    # Chunk objects are created by the indexing layer.
    # Here we verify the strategy's output can be converted
    # into identified Chunk records.

    from src.models.records import Chunk

    document = Document(
        doc_id="175.jpeg",
        content="A" * 25,
        qa_pairs=[]
    )

    chunker = RecursiveChunking(chunk_size=10, overlap=2)
    text_chunks = chunker.split(document.content)

    chunks = [
        Chunk(
            chunk_id=f"{document.doc_id}::chunk::{index}",
            doc_id=document.doc_id,
            text=text,
            chunk_index=index
        )
        for index, text in enumerate(text_chunks)
    ]

    assert len(chunks) > 0

    assert chunks[0].chunk_id == "175.jpeg::chunk::0"
    assert chunks[1].chunk_id == "175.jpeg::chunk::1"

    assert all(chunk.doc_id == "175.jpeg" for chunk in chunks)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

    assert all(isinstance(chunk, Chunk) for chunk in chunks)


def test_chunk_document_empty_content():
    document = Document(
        doc_id="empty_doc",
        content="   ",
        qa_pairs=[]
    )

    chunker = RecursiveChunking()

    assert chunker.split(document.content) == []
