
import chromadb
from sentence_transformers import SentenceTransformer
from src.pipeline.factory import _build_embedder
from src.config import AppConfig
import os
from pathlib import Path
import json

config = AppConfig.from_env()

CHROMA_PATH = config.chroma_path
COLLECTION_NAME = config.chroma_collection
#COLLECTION_NAME = "finance_sample_500_parent_child"
#DOC_ID = "175.jpeg"

print("Collection Name:",COLLECTION_NAME)
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)


# Use the same embedding model configured in your application.
#embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedder = _build_embedder(config)


def verify_chroma_collection():
    print("\n========== COLLECTION VERIFICATION ==========")

    print("Collection:", collection.name)
    print("Total stored chunks:", collection.count())

    results = collection.get(
        #where={"doc_id": DOC_ID},
        include=["documents", "metadatas"],
    )

    #print(f"\nChunks belonging to {DOC_ID}:")

    for chunk_id, metadata, content in zip(
        results["ids"],
        results["metadatas"],
        results["documents"],
    ):
        print("\nChunk ID:", chunk_id)
        print("Metadata:", metadata)
        print("Content:", content[:700])


def verify_retrieval_old():
    print("\n========== RETRIEVAL VERIFICATION ==========")

    query = "How much are personal services allocated in the budget?"

    query_embedding = [embedder.embed_query(query)]

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=10,
        #where={"doc_id": DOC_ID},
        include=["documents", "metadatas", "distances"],
    )
    

    print(f"\nRetrieval results for: {query}")
    #print(f"Document filter: {DOC_ID}")

    for rank, (chunk_id, metadata, distance, content) in enumerate(
        zip(
            result["ids"][0],
            result["metadatas"][0],
            result["distances"][0],
            result["documents"][0],
        ),
        start=1,
    ):
        similarity = 1.0 - float(distance)

        print(f"\nRank: {rank}")
        print("Chunk ID:", chunk_id)
        print("Metadata:", metadata)
        print(f"Distance: {distance:.4f}")
        print(f"Similarity: {similarity:.4f}")
        print("Content:", content[:300])



def verify_retrieval():
    print("\n========== RETRIEVAL VERIFICATION ==========")

    query = (
        "What is the location of Mallard Media Services?"
    )

    TARGET_DOC_ID = "3348.jpeg"
    TARGET_CHUNK_ID = "3348.jpeg:child:0:1"
    TOP_K = 100
    total_chunks = collection.count()
    search_limit = min(TOP_K, total_chunks)

    query_embedding = [embedder.embed_query(query)]

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=search_limit,
        include=["documents", "metadatas", "distances"],
    )

    retrieved_ids = result["ids"][0]
    retrieved_metadata = result["metadatas"][0]
    retrieved_distances = result["distances"][0]
    retrieved_documents = result["documents"][0]

    document_found = any(
        metadata.get("doc_id") == TARGET_DOC_ID
        for metadata in retrieved_metadata
    )

    chunk_found = (
        TARGET_CHUNK_ID in retrieved_ids
        if TARGET_CHUNK_ID is not None
        else None
    )

    matching_ranks = [
        rank
        for rank, metadata in enumerate(retrieved_metadata, start=1)
        if metadata.get("doc_id") == TARGET_DOC_ID
    ]

    # Save complete retrieval results to a file
    output_path = Path(
        "data/evaluation/500_docs_parent_child/test_retrieval_top100.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        file.write("========== GLOBAL RETRIEVAL VERIFICATION ==========\n")
        file.write(f"Query: {query}\n")
        file.write(f"Total indexed chunks: {total_chunks}\n")
        file.write(f"Requested TOP_K: {TOP_K}\n")
        file.write(f"Actual TOP_K: {search_limit}\n")
        file.write(f"Expected document: {TARGET_DOC_ID}\n")
        file.write(f"Correct document retrieved: {document_found}\n")
        file.write(f"Expected chunk: {TARGET_CHUNK_ID}\n")
        file.write(f"Exact target chunk retrieved: {chunk_found}\n")
        file.write(f"Matching document ranks: {matching_ranks}\n")
        file.write("\n========== ALL RETRIEVED CHUNKS ==========\n")

        for rank, (chunk_id, metadata, distance, content) in enumerate(
            zip(
                retrieved_ids,
                retrieved_metadata,
                retrieved_distances,
                retrieved_documents,
            ),
            start=1,
        ):
            similarity = 1.0 - float(distance)

            file.write(f"\nRank: {rank}\n")
            file.write(f"Chunk ID: {chunk_id}\n")
            file.write(f"Metadata: {metadata}\n")
            file.write(f"Distance: {distance:.4f}\n")
            file.write(f"Similarity: {similarity:.4f}\n")
            file.write(f"Content:\n{content}\n")
            file.write("-" * 80 + "\n")

    # Print only the summary in the terminal
    print(f"Query: {query}")
    print(f"Total indexed chunks: {total_chunks}")
    print(f"Results retrieved: {search_limit}")
    print(f"Correct document retrieved: {document_found}")
    print(f"Matching document ranks: {matching_ranks}")
    print(f"Exact target chunk retrieved: {chunk_found}")
    print(f"\nFull results saved to: {output_path}")


def verify_batch_retrieval():
    print("\n========== BATCH RETRIEVAL VERIFICATION ==========")

    # Path to evaluation gold set json (Parent-Child experiment)
    gold_set_path = Path("data/evaluation/500_docs_parent_child/evaluation_gold_set_200.json")
    
    if not gold_set_path.exists():
        print(f"Error: Gold set file not found at {gold_set_path}")
        return

    with gold_set_path.open("r", encoding="utf-8") as f:
        gold_data = json.load(f)

    # Output path for results
    output_path = Path("data/evaluation/500_docs_parent_child/global_retrieval_top100_today.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = collection.count()
    TOP_K = 100  # Optimized candidate pool for reranker evaluation
    search_limit = min(TOP_K, total_chunks)

    with output_path.open("w", encoding="utf-8") as file:
        for item in gold_data:
            query = item.get("question")
            TARGET_DOC_ID = item.get("doc_id")
            
            # Retrieve the first relevant chunk id if available (handling both key variants)
            relevant_chunks = item.get("relevant_chunk_id") or item.get("relevant_chunk_ids", [])
            TARGET_CHUNK_ID = relevant_chunks[0] if relevant_chunks else None

            if not query:
                continue

            query_embedding = [embedder.embed_query(query)]

            result = collection.query(
                query_embeddings=query_embedding,
                n_results=search_limit,
                include=["documents", "metadatas", "distances"],
            )
            print("Processed query:", query)

            retrieved_ids = result["ids"][0]
            retrieved_metadata = result["metadatas"][0]
            retrieved_distances = result["distances"][0]
            retrieved_documents = result["documents"][0]

            document_found = any(
                metadata.get("doc_id") == TARGET_DOC_ID
                for metadata in retrieved_metadata
            )

            chunk_found = (
                TARGET_CHUNK_ID in retrieved_ids
                if TARGET_CHUNK_ID is not None
                else False
            )

            matching_ranks = [
                rank
                for rank, metadata in enumerate(retrieved_metadata, start=1)
                if metadata.get("doc_id") == TARGET_DOC_ID
            ]

            # Write individual question summary to file
            file.write("========== GLOBAL RETRIEVAL VERIFICATION ==========\n")
            file.write(f"Query: {query}\n")
            file.write(f"question_id: {item.get('question_id')}\n")
            file.write(f"Gold Answer: {item.get('gold_answer')}\n")
            file.write(f"Total indexed chunks: {total_chunks}\n")
            file.write(f"Requested TOP_K: {TOP_K}\n")
            file.write(f"Actual TOP_K: {search_limit}\n")
            file.write(f"Expected document: {TARGET_DOC_ID}\n")
            file.write(f"Correct document retrieved: {document_found}\n")
            file.write(f"Expected chunk: {TARGET_CHUNK_ID}\n")
            file.write(f"Exact target chunk retrieved: {chunk_found}\n")
            file.write(f"Matching document ranks: {matching_ranks}\n")
            file.write("\n" + "=" * 50 + "\n\n")

    print(f"Processed {len(gold_data)} queries successfully.")
    print(f"Full results saved to: {output_path}")

def verify_target_global_rank():
    print("\n========== TARGET CHUNK GLOBAL RANK ==========")

    query = "What is the purpose of Recommendation No. 6?"
    target_chunk_id = "175.jpeg:1"

    query_embedding = [embedder.embed_query(query)]

    result = collection.query(
        query_embeddings=query_embedding,
        n_results=collection.count(),
        include=["documents", "metadatas", "distances"],
    )

    ranked_chunks = list(zip(
        result["ids"][0],
        result["metadatas"][0],
        result["distances"][0],
    ))

    target_rank = None

    for rank, (chunk_id, metadata, distance) in enumerate(
        ranked_chunks, start=1
    ):
        if chunk_id == target_chunk_id:
            target_rank = rank
            print("Target chunk:", chunk_id)
            print("Global rank:", rank)
            print(f"Similarity: {1.0 - float(distance):.4f}")
            print("Metadata:", metadata)
            break

    if target_rank is None:
        print("Target chunk was not found in the collection.")

    print("\nTop 10 global results:")
    for rank, (chunk_id, metadata, distance) in enumerate(
        ranked_chunks[:10], start=1
    ):
        print(
            f"Rank {rank}: {chunk_id} | "
            f"Similarity: {1.0 - float(distance):.4f}"
        )

    if target_rank is not None and target_rank > 10:
        print("\nChunks immediately surrounding the target:")
        start = max(0, target_rank - 4)
        end = min(len(ranked_chunks), target_rank + 3)

        for rank, (chunk_id, metadata, distance) in enumerate(
            ranked_chunks[start:end], start=start + 1
        ):
            print(
                f"Rank {rank}: {chunk_id} | "
                f"Similarity: {1.0 - float(distance):.4f}"
            )


def verify_collection_chunk_sizes(
    max_allowed: int = 350,
    labels_path: str | None = "data/evaluation/500_docs_parent_child/evaluation_gold_set_200.json",
) -> None:
    print("\n========== COLLECTION CHUNK SIZE CHECK ==========")
    print("Collection:", collection.name)
    print("Total stored chunks:", collection.count())

    results = collection.get(include=["documents", "metadatas"])
    ids = results["ids"]
    docs = results["documents"]
    metas = results["metadatas"]

    relevant_ids: set[str] = set()
    if labels_path and Path(labels_path).exists():
        raw = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        records = raw.get("questions", raw) if isinstance(raw, dict) else raw
        for rec in records:
            cids = rec.get("relevant_chunk_id") or []
            if isinstance(cids, str):
                cids = [cids]
            for cid in cids:
                if cid and str(cid).strip():
                    relevant_ids.add(str(cid).strip())

    all_char_lens: list[int] = []
    all_word_lens: list[int] = []
    over_350_all: list[tuple[int, str, int, int]] = []
    rel_char_lens: list[int] = []
    over_350_rel: list[tuple[int, str, int]] = []

    for cid, meta, content in zip(ids, metas, docs):
        text = content or ""
        c_len = len(text)
        w_len = len(text.split())
        all_char_lens.append(c_len)
        all_word_lens.append(w_len)

        if c_len > max_allowed:
            over_350_all.append((c_len, w_len, cid, (meta or {}).get("chunk_type")))

        if cid in relevant_ids:
            rel_char_lens.append(c_len)
            if c_len > max_allowed:
                over_350_rel.append((c_len, cid, text[:100]))

    if all_char_lens:
        print(
            f"ALL chunks ({len(all_char_lens)}) -> "
            f"min={min(all_char_lens)} chars, avg={sum(all_char_lens)/len(all_char_lens):.1f} chars, "
            f"max={max(all_char_lens)} chars (max words={max(all_word_lens)})"
        )
        print(f"ALL chunks > {max_allowed} chars: {len(over_350_all)}")
        for c_len, w_len, cid, ctype in sorted(over_350_all, reverse=True)[:10]:
            print(f"  id={cid} | type={ctype} | chars={c_len} | words={w_len}")

    if relevant_ids:
        print(f"\nLabeled relevant IDs in gold set: {len(relevant_ids)} | Found in collection: {len(rel_char_lens)}")
        if rel_char_lens:
            print(
                f"LABELED chunks -> min={min(rel_char_lens)} chars, "
                f"avg={sum(rel_char_lens)/len(rel_char_lens):.1f} chars, max={max(rel_char_lens)} chars"
            )
            print(f"LABELED chunks > {max_allowed} chars: {len(over_350_rel)}")
            for c_len, cid, preview in sorted(over_350_rel, reverse=True)[:10]:
                print(f"  id={cid} | chars={c_len} | preview={preview!r}")


def check():
    config = AppConfig.from_env()
    print("\n\nConfig:",config)
    chroma_dir = Path(config.chroma_path)
    if not chroma_dir.is_absolute():
        chroma_dir = Path.cwd() / chroma_dir

    print("Resolved CHROMA_PATH:", chroma_dir, "| exists:", chroma_dir.exists())

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collections = client.list_collections()
    print("Existing collections:", [c.name for c in collections])




if __name__ == "__main__":
    #verify_chroma_collection()
    #verify_retrieval()
    verify_batch_retrieval()
    #verify_target_global_rank()
    #verify_retrieval_old()
    #verify_collection_chunk_sizes()
    #check()
