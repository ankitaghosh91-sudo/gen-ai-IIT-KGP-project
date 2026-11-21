# RAG Retrieval Evaluation Summary (`500_docs_parent_child`)

## 1. Challenges Encountered
* **Misleading 14-line output & wrong sort key:** Early inspection showed only 14 lines in lexicographical order (`qq1`, `qq10`, `qq11`...), making it look like the evaluation stopped early or got truncated when the issue was in how records were indexed and printed.
* **Indexing `question_id` instead of `question_number`:** String-sorting `question_id` distorted numeric question order and caused awkward `q`-prefix formatting (`qqq1`), hiding the full 180-question view.
* **Diagnosing end-to-end vs. stage-specific misses:** With 180 questions across four stages (`Dense (Chroma)`, `Sparse (BM25)`, `Hybrid`, and `Hybrid + BGE Reranker`), it wasn't immediately obvious whether failures came from upstream retrieval, hybrid fusion/candidate cutoffs, reranker drop-offs, or chunk granularity.

---

## 2. Metrics & Failure Breakdown (180 Questions)

| Metric / Bucket | Count (out of 180) | Share of Total (180) | Share of Failing / ≥1 `None` (103) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Total Questions Evaluated** | 180 | 100.00% | — | Numerically sorted `q1`–`q180` |
| **Any Strategy Hit (≥1 of 4 non-`None`)** | 115 | 63.89% | — | At least one stage returned a relevant chunk |
| **Final `Hybrid + BGE Reranker` Hit** | 100 | 55.56% | — | Final stage returned a relevant chunk |
| **All Four Strategies Hit (0 `None`s)** | 77 | 42.78% | — | Dense, Sparse, Hybrid, and Reranker all hit |
| **Questions with ≥1 Strategy `None`** | 103 | 57.22% | 100.00% | Total questions in the failure breakdown |
| `all_strategies_none` | 65 | 36.11% | 63.11% | Missed across all 4 stages; includes sibling `:child:` hits & generic/inference queries |
| `found_earlier_lost_at_reranker` | 15 | 8.33% | 14.56% | Single-retriever hit at ranks 22–39; `Hybrid` (`retrieved=40`) already `None` before reranking |
| `partial_or_other_miss` | 23 | 12.78% | 22.33% | At least one earlier strategy missed (`None`), while `reranker` still found a match |
| `found_in_base_lost_in_hybrid` | 0 | 0.00% | 0.00% | Caught under `found_earlier_lost_at_reranker` due to `if/elif` ordering |
| `reranker_none` | 0 | 0.00% | 0.00% | Caught under `found_earlier_lost_at_reranker` due to `if/elif` ordering |

*(Note: Percentages above measure **hit presence**—whether a relevant chunk appeared anywhere in the retrieved list of 40—not top-K precision, recall@K, or MRR.)*

---

## 3. Key Findings by Bucket
* **`all_strategies_none` (65 queries / 36.11%):** None of the four stages matched the gold child chunk ID. A 10-query inspection revealed two clear patterns:
  1. **Sibling child-chunk misses:** The right document (`doc_id`) and a neighboring `:child:` chunk appeared near the top (`#1`–`#2`), such as `3229.jpeg:child:0:4` vs. gold `3229.jpeg:child:1:1` (`q9`), `8065.jpeg:child:0:4` vs. `8065.jpeg:child:1:1` (`q11`), `4254.jpeg:child:0:0` vs. `4254.jpeg:child:0:1` (`q22`), and `6080.jpeg:child:1:2` vs. `6080.jpeg:child:2:2` (`q4`).
  2. **Unanchored / generic questions & analytical inference:** Generic questions without entity or document anchors (e.g., *"Who is the issuer of the invoice?"*, *"What is the purpose of the payment?"*, *"What is the payment term for the Purchase Order?"*) collide across similar financial documents, while inference/synthesis questions may need broader parent or multi-chunk context.
* **`found_earlier_lost_at_reranker` (15 queries / 8.33%):** Every record in this bucket had a single-retriever hit deep in the tail (**ranks 22–39**: 6 in Dense, 9 in Sparse) while the other retriever returned `None`. In all 15 cases, `Hybrid` (`retrieved=40`) already returned `None` before the reranker ran.
* **`partial_or_other_miss` (23 queries / 12.78%):** At least one individual strategy returned `None`, while `Hybrid + BGE Reranker` still succeeded.
* **Positive lift & complementarity:** Where hits are surfaced, `Hybrid` and `Hybrid + BGE Reranker` frequently pull mid-ranked candidates into top positions (e.g., `q1`, `q2`, `q18`, `q65`), and `Sparse (BM25)` rescues keyword-heavy queries missed by `Dense (Chroma)` (e.g., `q36`, `q49`, `q59`, `q162`).

---

## 4. Steps Taken & Scripts Built
1. **`rank_comparison_report.py`:** Fixed indexing and sorting to use numeric `question_number` (with digit fallback), included `question_id` for traceability, added input/directory checks, and generated `rank_comparison_report.txt` across all 180 queries.
2. **`failure_analysis.py`:** Categorized all 103 failing queries into failure buckets and exported `failure_analysis_report.json`.
3. **`inspect_reranker_losses.py`:** Inspected all 15 `found_earlier_lost_at_reranker` records with `retrieved=40` and hit rank positions.
4. **`inspect_all_none.py`:** Inspected a sample of `all_strategies_none` records (10 of 65) with top-3 retrieved chunk IDs to diagnose root causes.

---

## 5. What Is Still Unhandled in the Pipeline
* **Remaining `all_strategies_none` records (55 of 65 uninspected):** Quantifying how many of the 65 queries contain a sibling `:child:` chunk from the target `doc_id` versus missing the document altogether.
* **Parent vs. child evaluation & context alignment:** Checking how exact child-chunk ID matching affects apparent recall on parent-child chunks, and whether parent/multi-chunk context helps inference questions.
* **Hybrid fusion & candidate pool behavior (`top_k` / RRF / weights):** Inspecting how `Hybrid` constructs its top-40 candidate pool and why single-retriever hits at ranks 22–39 are absent from `Hybrid`'s top-40 list.
* **Reranker regressions on existing hits:** Diagnosing cases where `Hybrid` or a base retriever ranked a relevant chunk higher than `Hybrid + BGE Reranker` (e.g., `q43`, `q60`, `q171`).
* **Top-K ranking metrics (`Precision@K`, `Recall@K`, `MRR`):** Evaluating rank quality (such as rank-1 hit rate) beyond binary presence in the top-40 retrieved list.