
"""Save retrieval evaluation results to a JSON file."""

import json
from pathlib import Path


DEFAULT_OUTPUT_PATH = (
    "data/evaluation/retrieval_evaluation_results_withBGEEmbedder.json"
)


def save_evaluation_results(
    evaluation_results: dict,
    averages: dict,
    total_qa_pairs: int,
    eligible_qa_pairs: int,
    excluded_qa_pairs: int,
    output_path: str = DEFAULT_OUTPUT_PATH,
):
    output_path = Path(output_path)

    report = {
        "evaluation_summary": {
            "total_qa_pairs": total_qa_pairs,
            "eligible_qa_pairs": eligible_qa_pairs,
            "excluded_qa_pairs": excluded_qa_pairs,
        },
        "average_metrics": averages,
        "per_question_metrics": evaluation_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"\nEvaluation report saved to: {output_path}")


if __name__ == "__main__":
    print("This module is intended to be called from the evaluation script.")
