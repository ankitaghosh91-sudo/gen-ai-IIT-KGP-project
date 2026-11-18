
import argparse
import json
from pathlib import Path


DEFAULT_EXACT = Path("data/evaluation/relevance_labels.json")
DEFAULT_LLM = Path("data/evaluation/llm_relevance_labels.json")
DEFAULT_OUTPUT = Path("data/evaluation/combined_relevance_labels.json")


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def qa_key(record):
    return (record["doc_id"], record["question_id"])


def merge_labels(exact_labels, llm_labels):
    # Group LLM judgments by QA pair.
    llm_by_qa = {}

    for record in llm_labels:
        llm_by_qa.setdefault(qa_key(record), []).append(record)

    combined_labels = []

    for exact_record in exact_labels:
        key = qa_key(exact_record)
        exact_relevant = exact_record["relevant_chunks"]

        # Start with exact-match relevance labels.
        combined = {
            "doc_id": exact_record["doc_id"],
            "question_id": exact_record["question_id"],
            "question": exact_record["question"],
            "gold_answer": exact_record["gold_answer"],
            "relevant_chunk_ids": list(exact_relevant),
            "partially_relevant_chunk_ids": [],
            "irrelevant_chunk_ids": [],
            "uncertain_chunk_ids": [],
            "label_source": "exact_match",
            "candidate_judgments": [],
        }

        # If exact matching found no relevant chunk,
        # use the LLM judgments for this QA pair.
        if not exact_relevant:
            candidates = llm_by_qa.get(key, [])

            if not candidates:
                raise ValueError(
                    f"No LLM judgments found for QA pair: {key}"
                )

            combined["label_source"] = "llm_judgment"

            for candidate in candidates:
                if candidate.get("error") is not None:
                    raise ValueError(
                        f"LLM judgment error for {key}: "
                        f"{candidate['error']}"
                    )

                chunk_id = candidate["chunk_id"]
                label = candidate["llm_label"]

                if label == "relevant":
                    combined["relevant_chunk_ids"].append(chunk_id)

                elif label == "partially_relevant":
                    combined["partially_relevant_chunk_ids"].append(
                        chunk_id
                    )

                elif label == "irrelevant":
                    combined["irrelevant_chunk_ids"].append(chunk_id)

                elif label == "uncertain":
                    combined["uncertain_chunk_ids"].append(chunk_id)

                else:
                    raise ValueError(
                        f"Unknown LLM label: {label}"
                    )

                # Preserve the original LLM judgment details.
                combined["candidate_judgments"].append(candidate)

        combined_labels.append(combined)

    return combined_labels


def generate_report(combined_labels):
    print("\n========== MERGE SUMMARY ==========")
    print("Combined QA pairs:", len(combined_labels))

    print(
        "Exact-match QA pairs:",
        sum(
            r["label_source"] == "exact_match"
            for r in combined_labels
        ),
    )

    print(
        "LLM-judged QA pairs:",
        sum(
            r["label_source"] == "llm_judgment"
            for r in combined_labels
        ),
    )

    print(
        "QA pairs with at least one relevant chunk:",
        sum(
            bool(r["relevant_chunk_ids"])
            for r in combined_labels
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Merge exact-match and LLM relevance labels."
    )

    parser.add_argument(
        "--exact",
        type=Path,
        default=DEFAULT_EXACT,
        help="Path to exact-match relevance labels.",
    )

    parser.add_argument(
        "--llm",
        type=Path,
        default=DEFAULT_LLM,
        help="Path to LLM relevance labels.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path for the combined relevance labels.",
    )

    args = parser.parse_args()

    exact_labels = load_json(args.exact)
    llm_labels = load_json(args.llm)

    combined_labels = merge_labels(exact_labels, llm_labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            combined_labels,
            f,
            indent=2,
            ensure_ascii=False,
        )

    generate_report(combined_labels)
    print("Saved to:", args.output)


if __name__ == "__main__":
    main()
