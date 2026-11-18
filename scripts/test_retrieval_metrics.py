
from src.evaluation.metrics import evaluate_query

retrieved = [
    "doc1:chunk1",
    "doc1:chunk2",
    "doc1:chunk3",
    "doc2:chunk1",
    "doc2:chunk2",
]

relevant = {
    "doc1:chunk2",
    "doc2:chunk2",
}

results = evaluate_query(retrieved, relevant)

print("Retrieved:", retrieved)
print("Relevant:", relevant)

print("\nEvaluation metrics:")

for metric, score in results.items():
    print(f"{metric}: {score:.4f}")
