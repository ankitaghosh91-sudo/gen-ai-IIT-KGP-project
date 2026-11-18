
from src.evaluation.relevance_judge import RelevanceJudge


def test_relevance_judge():

    judge = RelevanceJudge()

    question = (
        "What is the potential consequence of not "
        "cross-referencing this document with campaign outcomes?"
    )

    expected_answer = (
        "The potential consequence could be a lack of "
        "evaluation on financial efficiency and ROI."
    )


    candidate_chunk = """
        "chunk_text": "This document is an essential part of financial and compliance archival for the advertising campaign's expenditure tracking. It could be cross-referenced with campaign outcomes to evaluate financial efficiency and ROI."
    """

    result = judge.judge(
        question=question,
        expected_answer=expected_answer,
        candidate_chunk=candidate_chunk,
    )

    print("Relevance Judge Result")
    print("----------------------")
    print("Label:", result["label"])
    print("Reason:", result["reason"])


if __name__ == "__main__":
    test_relevance_judge()
