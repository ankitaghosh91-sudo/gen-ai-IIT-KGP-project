import json
from pathlib import Path
from src.data import load_saved_sample
from src.config import AppConfig
from src.pipeline.factory import _build_chunker
from src.strategies.search import BM25Search


def test_bm25_retrieval():
    gold_set_path = Path("data/evaluation/500_docs_parent_child/evaluation_gold_set_200.json")
    
    if not gold_set_path.exists():
        print(f"Error: Gold set file not found at {gold_set_path}")
        return

    with gold_set_path.open("r", encoding="utf-8") as f:
        gold_data = json.load(f)

    # FIX: Use the exact same 500-document sample path as ChromaDB indexing
    INPUT_PATH = "data/sample_500/documents.json"
    documents = load_saved_sample(path=INPUT_PATH)

    if not documents:
        print("No documents found.")
        return

    config = AppConfig.from_env()
    chunker = _build_chunker(config)

    search = BM25Search(chunker=chunker)

    max_queries = 200
    eval_subset = gold_data[:max_queries]

    output_path = Path("data/evaluation/500_docs_parent_child/bm25_retrieval_results.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n========== BM25 BATCH RETRIEVAL EVALUATION ==========")
    print(f"Loaded {len(eval_subset)} queries from gold set.")
    print(f"Loaded {len(documents)} documents for BM25 corpus.")

    doc_correct_count = 0
    chunk_correct_count = 0
    total_processed = 0

    with output_path.open("w", encoding="utf-8") as file:
        for item in eval_subset:
            query = item.get("question")
            target_doc_id = item.get("doc_id")
            
            relevant_chunks = item.get("relevant_chunk_id") or item.get("relevant_chunk_ids", [])
            target_chunk_id = relevant_chunks[0] if relevant_chunks else None

            if not query:
                continue

            total_processed += 1
            TOP_K = 100

            results = search.search(
                query=query,
                documents=documents,
                limit=TOP_K,
            )

            retrieved_chunk_ids = [res.chunk_id for res in results]

            # Safe normalization for retrieved doc IDs and matching ranks
            retrieved_doc_ids = []
            matching_ranks = []
            
            for rank, res in enumerate(results, start=1):
                d_id = res.doc_id
                # Handle list vs string doc_ids cleanly
                if isinstance(d_id, list):
                    normalized_ids = [str(x).strip() for x in d_id]
                    retrieved_doc_ids.extend(normalized_ids)
                    if target_doc_id in normalized_ids:
                        matching_ranks.append(rank)
                else:
                    norm_id = str(d_id).strip()
                    retrieved_doc_ids.append(norm_id)
                    if norm_id == target_doc_id:
                        matching_ranks.append(rank)

            document_found = target_doc_id in retrieved_doc_ids
            chunk_found = target_chunk_id in retrieved_chunk_ids if target_chunk_id else False

            if document_found:
                doc_correct_count += 1
            if chunk_found:
                chunk_correct_count += 1

            # Write individual query summary to output log
            file.write("========== BM25 RETRIEVAL VERIFICATION ==========\n")
            file.write(f"Query: {query}\n")
            file.write(f"Question ID: {item.get('question_id')}\n")
            file.write(f"Expected Document: {target_doc_id}\n")
            file.write(f"Correct Document Retrieved: {document_found}\n")
            file.write(f"Expected Chunk: {target_chunk_id}\n")
            file.write(f"Exact Target Chunk Retrieved: {chunk_found}\n")
            file.write(f"Matching Document Ranks: {matching_ranks}\n")
            file.write("\n" + "=" * 50 + "\n\n")

            print(f"Processed [{total_processed}/{len(eval_subset)}] | Doc Found: {document_found} | Ranks: {matching_ranks}")

    doc_accuracy = (doc_correct_count / total_processed) * 100 if total_processed > 0 else 0
    chunk_accuracy = (chunk_correct_count / total_processed) * 100 if total_processed > 0 else 0

    print("\n========== BM25 EVALUATION SUMMARY ==========")
    print(f"Total Queries Processed: {total_processed}")
    print(f"Document-Level Accuracy: {doc_accuracy:.2f}% ({doc_correct_count}/{total_processed})")
    print(f"Chunk-Level Accuracy: {chunk_accuracy:.2f}% ({chunk_correct_count}/{total_processed})")
    print(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    test_bm25_retrieval()