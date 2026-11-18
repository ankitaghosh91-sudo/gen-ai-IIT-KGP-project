
"""Retrieval evaluation metrics."""

def precision_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    """Calculate Precision@K."""

    if k < 1:
        raise ValueError("k must be positive")

    top_k = retrieved_chunk_ids[:k]

    relevant_count = sum(
        chunk_id in relevant_chunk_ids
        for chunk_id in top_k
    )

    return relevant_count / k


def recall_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    """Calculate Recall@K."""

    if k < 1:
        raise ValueError("k must be positive")

    if not relevant_chunk_ids:
        return 0.0

    top_k = retrieved_chunk_ids[:k]

    relevant_count = sum(
        chunk_id in relevant_chunk_ids
        for chunk_id in top_k
    )

    return relevant_count / len(relevant_chunk_ids)


def reciprocal_rank_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    """Calculate Reciprocal Rank at K."""

    if k < 1:
        raise ValueError("k must be positive")

    for rank, chunk_id in enumerate(retrieved_chunk_ids[:k], start=1):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank

    return 0.0


def evaluate_query(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k_values: list[int] = [1, 3, 5],
) -> dict:
    """Calculate retrieval metrics for one QA pair."""

    return {
        f"precision@{k}": precision_at_k(
            retrieved_chunk_ids, relevant_chunk_ids, k
        )
        for k in k_values
    } | {
        f"recall@{k}": recall_at_k(
            retrieved_chunk_ids, relevant_chunk_ids, k
        )
        for k in k_values
    } | {
        f"mrr@{k}": reciprocal_rank_at_k(
            retrieved_chunk_ids, relevant_chunk_ids, k
        )
        for k in k_values
    }
