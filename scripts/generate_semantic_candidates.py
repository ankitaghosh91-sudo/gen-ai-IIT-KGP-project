
"""Generate semantic candidate chunks for unmatched RAG evaluation labels."""

import argparse
import json
from pathlib import Path

from src.config import AppConfig
from src.data import load_saved_sample
from src.pipeline.factory import _build_chunker, _build_embedder
from src.strategies.helpers import cosine_similarity
from src.strategies.search import build_chunks


TOP_K = 3

DEFAULT_INPUT = "data/sample/documents.json"
DEFAULT_LABELS = "data/evaluation/relevance_labels.json"
DEFAULT_OUTPUT = "data/evaluation/semantic_candidates.json"


def generate_semantic_candidates(
    input_path: str,
    labels_path: str,
    output_path: str,
):
    # 1. Load configuration and initialize strategies.
    config = AppConfig.from_env()

    chunker = _build_chunker(config)
    embedder = _build_embedder(config)

    # 2. Load documents and existing exact-match labels.
    documents = load_saved_sample(input_path)

    if not documents:
        raise ValueError(
            f"No saved documents found at: {input_path}"
        )

    labels_file = Path(labels_path)

    if not labels_file.exists():
        raise FileNotFoundError(
            f"Relevance labels not found: {labels_file}"
        )

    labels = json.loads(labels_file.read_text(encoding="utf-8"))

    if not isinstance(labels, list):
        raise ValueError("Relevance labels must contain a JSON list.")

    # 3. Index labels by document and question ID.
    labels_by_key = {}

    for label in labels:
        key = (label["doc_id"], label["question_id"])

        if key in labels_by_key:
            raise ValueError(f"Duplicate relevance label: {key}")

        labels_by_key[key] = label

    candidate_records = []
    processed_questions = 0

    # 4. Process each document independently.
    for doc_index, document in enumerate(documents, start=1):

        unmatched_questions = []

        for qa_index, qa in enumerate(document.qa_pairs):
            question_id = f"q{qa_index + 1}"
            key = (document.doc_id, question_id)

            if key not in labels_by_key:
                raise ValueError(f"Missing relevance label: {key}")

            label = labels_by_key[key]

            if not label["relevant_chunks"]:
                unmatched_questions.append({
                    "question_id": question_id,
                    "question": qa.question,
                    "gold_answer": qa.answer,
                })

        if not unmatched_questions:
            continue

        # 5. Build chunks using the same configured chunker as RAG.
        chunks = build_chunks([document], chunker)

        if not chunks:
            continue

        # 6. Embed all chunks once for this document.
        chunk_embeddings = embedder.embed_texts(
            [chunk.text for chunk in chunks]
        )
        if len(chunk_embeddings) != len(chunks):
            raise ValueError(
                f"Expected {len(chunks)} chunk embeddings, "
                f"received {len(chunk_embeddings)}"
            )

        # 7. Prepare all query texts for this document.
        candidate_queries = [
            (
                f"Question: {question['question']}\n"
                f"Expected answer: {question['gold_answer']}"
            )
            for question in unmatched_questions
        ]

        # 8. Embed queries in batches of up to 16.
        query_embeddings = embedder.embed_queries(candidate_queries)
        if len(query_embeddings) != len(unmatched_questions):
            raise ValueError(
                f"Expected {len(unmatched_questions)} query embeddings, "
                f"received {len(query_embeddings)}"
        )

        # 9. Rank chunks separately for each unmatched QA pair.
        for question, query_embedding in zip(
            unmatched_questions,
            query_embeddings,
        ):
            scored_chunks = []

            for chunk, chunk_embedding in zip(chunks, chunk_embeddings):

                score = cosine_similarity(
                    query_embedding,
                    chunk_embedding,
                )

                scored_chunks.append({
                    "chunk_id": chunk.chunk_id,
                    "similarity_score": round(float(score), 6),
                    "chunk_text": chunk.text,
                })

            top_candidates = sorted(
                scored_chunks,
                key=lambda item: item["similarity_score"],
                reverse=True,
            )[:TOP_K]

            candidate_records.append({
                "doc_id": document.doc_id,
                "question_id": question["question_id"],
                "question": question["question"],
                "gold_answer": question["gold_answer"],
                "candidates": top_candidates,
                "label_status": "pending_validation",
            })

            processed_questions += 1

    if doc_index % 50 == 0 or doc_index == len(documents):
        print(
            f"Processed {doc_index}/{len(documents)} documents | "
            f"Candidate records: {processed_questions}"
        )

    # 9. Save candidates separately. Original labels remain untouched.
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_file.write_text(
        json.dumps(candidate_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== SEMANTIC CANDIDATE SUMMARY ==========")
    print("Documents processed:", len(documents))
    print("Unmatched QA pairs processed:", processed_questions)
    print("Candidate chunks per QA pair:", TOP_K)
    print("Candidate records generated:", len(candidate_records))
    print("Candidates saved to:", output_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate semantic candidates for unmatched QA labels."
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to the saved documents.json manifest.",
    )

    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS,
        help="Path to the exact-match relevance labels.",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path where semantic candidates will be saved.",
    )

    args = parser.parse_args()

    generate_semantic_candidates(
        input_path=args.input,
        labels_path=args.labels,
        output_path=args.output,
    )
