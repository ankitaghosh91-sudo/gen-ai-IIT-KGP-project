
import json
from pathlib import Path
from collections import Counter

EVALUATION_DIR = Path("data/evaluation")

with open(EVALUATION_DIR / "relevance_labels.json", encoding="utf-8") as f:
    exact_labels = json.load(f)

with open(EVALUATION_DIR / "llm_relevance_labels.json", encoding="utf-8") as f:
    llm_labels = json.load(f)


def qa_key(record):
    return (record["doc_id"], record["question_id"])


exact_keys = {qa_key(record) for record in exact_labels}
llm_keys = {qa_key(record) for record in llm_labels}

print("===== FILE COUNTS =====")
print("Exact label records:", len(exact_labels))
print("Unique exact QA pairs:", len(exact_keys))
print("LLM candidate records:", len(llm_labels))
print("Unique LLM QA pairs:", len(llm_keys))

print("\n===== QA PAIR COVERAGE =====")
print("QA pairs in both files:", len(exact_keys & llm_keys))
print("Only in exact labels:", len(exact_keys - llm_keys))
print("Only in LLM labels:", len(llm_keys - exact_keys))
print("Total unique QA pairs:", len(exact_keys | llm_keys))

print("\n===== EXACT LABEL SUMMARY =====")
empty_exact = sum(
    not record["relevant_chunks"]
    for record in exact_labels
)
print("QA pairs with exact relevant chunks:", len(exact_labels) - empty_exact)
print("QA pairs without exact relevant chunks:", empty_exact)

print("\n===== LLM LABEL SUMMARY =====")
label_counts = Counter(
    record["llm_label"] for record in llm_labels
)
print("Judgment counts:", dict(label_counts))

llm_errors = sum(
    record.get("error") is not None
    for record in llm_labels
)
print("LLM records with errors:", llm_errors)

print("\n===== CANDIDATES PER QA PAIR =====")
candidate_counts = Counter(
    qa_key(record) for record in llm_labels
)
print("Candidate count distribution:", dict(Counter(candidate_counts.values())))
