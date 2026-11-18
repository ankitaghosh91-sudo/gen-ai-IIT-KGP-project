"""Generate a stratified 200-QA evaluation gold set from combined relevance labels."""

import argparse
import json
import random
from pathlib import Path

DEFAULT_INPUT = Path("data/evaluation/500_docs_parent_child/combined_relevance_labels.json")
DEFAULT_OUTPUT = Path("data/evaluation/500_docs_parent_child/evaluation_gold_set_200.json")


def generate_gold_set(input_path: Path, output_path: Path, target_size: int = 200, seed: int = 42):
    if not input_path.exists():
        raise FileNotFoundError(f"Combined relevance labels not found: {input_path}")

    records = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Combined relevance labels must be a JSON list.")

    print(f"Loaded {len(records)} total QA pairs from {input_path}")

    # Separate records into strata to ensure balanced representation
    exact_match_pool = [r for r in records if r.get("label_source") == "exact_match"]
    llm_relevant_pool = [
        r for r in records 
        if r.get("label_source") == "llm_judgment" and bool(r.get("relevant_chunk_ids"))
    ]
    llm_other_pool = [
        r for r in records 
        if r.get("label_source") == "llm_judgment" and not bool(r.get("relevant_chunk_ids"))
    ]

    print(f"Pool sizes -> Exact Match: {len(exact_match_pool)}, LLM Relevant: {len(llm_relevant_pool)}, LLM Partial/Other: {len(llm_other_pool)}")

    random.seed(seed)

    # Allocate samples proportionally (~50% exact match, ~40% LLM relevant, ~10% partial/other)
    sample_exact = min(len(exact_match_pool), int(target_size * 0.50))
    sample_llm_rel = min(len(llm_relevant_pool), int(target_size * 0.40))
    sample_other = target_size - (sample_exact + sample_llm_rel)

    selected_records = (
        random.sample(exact_match_pool, sample_exact) +
        random.sample(llm_relevant_pool, sample_llm_rel) +
        random.sample(llm_other_pool, min(sample_other, len(llm_other_pool)))
    )

    if len(selected_records) < target_size:
        remaining_pool = [r for r in records if r not in selected_records]
        additional_needed = target_size - len(selected_records)
        selected_records.extend(random.sample(remaining_pool, min(additional_needed, len(remaining_pool))))

    random.shuffle(selected_records)

    gold_set_output = []

    for idx, record in enumerate(selected_records, start=1):
        eval_id = f"eval_{idx:03d}"
        doc_id = record["doc_id"]
        question_id = record["question_id"]
        question = record["question"]
        gold_answer = record["gold_answer"]
        label_source = record.get("label_source", "unknown")

        relevant_chunk_ids = record.get("relevant_chunk_ids", [])
        partially_relevant_chunk_ids = record.get("partially_relevant_chunk_ids", [])

        candidate_judgments = record.get("candidate_judgments", [])
        chunk_metrics = {}
        relevant_chunk_ranks = []
        relevant_chunk_similarity_scores = []

        for c in candidate_judgments:
            c_id = c["chunk_id"]
            c_rank = c.get("rank")
            c_score = c.get("similarity_score")
            
            chunk_metrics[c_id] = {
                "rank": c_rank,
                "similarity_score": c_score,
                "llm_label": c.get("llm_label")
            }

            if c_id in relevant_chunk_ids:
                if c_rank is not None:
                    relevant_chunk_ranks.append(c_rank)
                if c_score is not None:
                    relevant_chunk_similarity_scores.append(c_score)

        # Fallback rank and similarity score for primary evaluation sorting
        primary_rank = relevant_chunk_ranks[0] if relevant_chunk_ranks else (candidate_judgments[0]["rank"] if candidate_judgments else None)
        primary_score = relevant_chunk_similarity_scores[0] if relevant_chunk_similarity_scores else (candidate_judgments[0]["similarity_score"] if candidate_judgments else None)

        gold_set_output.append({
            "eval_id": eval_id,
            "doc_id": doc_id,
            "question_id": question_id,
            "question": question,
            "gold_answer": gold_answer,
            "relevant_chunk_id": relevant_chunk_ids,
            "partially_relevant_chunk_id": partially_relevant_chunk_ids,
            "rank": primary_rank,
            "similarity_score": primary_score,
            "relevant_chunk_ranks": relevant_chunk_ranks,
            "relevant_chunk_similarity_scores": relevant_chunk_similarity_scores,
            "chunk_metrics": chunk_metrics,
            "label_source": label_source,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(gold_set_output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== GOLD SET SUMMARY ==========")
    print(f"Total evaluation gold records generated: {len(gold_set_output)}")
    print(f"Stratified sample breakdown -> Exact Match: {sample_exact}, LLM Relevant: {sample_llm_rel}, Other/Partial: {sample_other}")
    print(f"Gold set saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stratified 200-QA evaluation gold set.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to combined relevance labels")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to save evaluation gold set")
    parser.add_argument("--size", type=int, default=200, help="Target size of evaluation gold set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    generate_gold_set(args.input, args.output, target_size=args.size, seed=args.seed)