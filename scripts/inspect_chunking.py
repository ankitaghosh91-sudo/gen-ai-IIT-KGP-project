from src.data import load_saved_sample
from src.config import AppConfig
from src.strategies.chunking import RecursiveChunking
from src.pipeline.factory import _build_chunker
from src.strategies.search import build_chunks


def inspect_real_documents():

    # 1. Load the already saved finance documents
    documents = load_saved_sample()

    doc = documents[0]
    print("First Document")
    print("Document ID:", doc.doc_id)
    print("Content type:", type(doc.content))
    print("Content length:", len(doc.content))
    print("Content preview:", doc.content[:500])

    print("\nQA pairs:", len(doc.qa_pairs))
    print("Image path:", doc.image_path)

    for index, qa in enumerate(doc.qa_pairs, start=1):
        print(f"\n\n Question {index}: {qa.question}")
        print(f"\n Answer {index}: {qa.answer}")

    if not documents:
        print("No documents found.")
        return

    # 2. Initialize our recursive chunking strategy
    config = AppConfig.from_env()
    chunker = _build_chunker(config)

    total_chunks = 0

    # 3. Chunk each actual document
    for document in documents[:3]:

        # chunks = chunker.split(document.content)
        chunks = build_chunks([document], chunker)

        for index in range(len(chunks) - 1):
            current_chunk = chunks[index]
            next_chunk = chunks[index + 1]

            max_overlap = min(len(current_chunk.text), len(next_chunk.text))

            actual_overlap = 0

            for size in range(max_overlap, 0, -1):
                if current_chunk.text[-size:] == next_chunk.text[:size]:
                    actual_overlap = size
                    break

            print(
                f"Overlap between chunk {index} and {index + 1}: "
                f"{actual_overlap} characters"
            )

            if actual_overlap == 0:
                print("=" * 90)
                print(f"Document ID: {document.doc_id} | Zero Overlap Between Chunk {index} and {index + 1}")

                
                print(f"\n[Chunk {index} ID: {current_chunk.chunk_id}] (Length: {len(current_chunk.text)} chars)")
                print("-" * 50)
                print(current_chunk.text)
                print("-" * 50)
                
                print(f"\n[Chunk {index + 1} ID: {next_chunk.chunk_id}] (Length: {len(next_chunk.text)} chars)")
                print("-" * 50)
                print(next_chunk.text)
                print("-" * 50 + "\n")
           

        print("\n" + "=" * 90)
        print(f"Document ID: {document.doc_id}")
        print(f"Original content length: {len(document.content)}")
        print(f"Number of QA pairs: {len(document.qa_pairs)}")
        print(f"Number of chunks: {len(chunks)}")

        total_chunks += len(chunks)

        # 4. Inspect individual chunks
        for index, chunk in enumerate(chunks):

            print(f"\nChunk {index}")
            print(f"Characters: {len(chunk.text)}")
            print(f"Content: {chunk.text[:300]}")

            assert isinstance(chunk.text, str)
            assert chunk.text.strip()
            assert len(chunk.text) <= 800

    print("\n" + "=" * 90)
    print(f"Total documents: {len(documents)}")
    print(f"Total chunks generated: {total_chunks}")


if __name__ == "__main__":
    inspect_real_documents()
