import json
from collections import Counter
from pathlib import Path



INPUT_FILE = Path("data/evaluation/500_docs_parent_child/retrieval_evaluation_results_with_reranker.json")

def quick_schema_check():
    data = json.loads(Path("data/evaluation/500_docs_parent_child/retrieval_evaluation_results_with_reranker.json").read_text(encoding="utf-8"))
    print("Top keys:", list(data.keys()))
    detailed = data.get("detailed_results", {})
    print("Detailed type & keys:", type(detailed), list(detailed.keys()) if isinstance(detailed, dict) else len(detailed))
    if isinstance(detailed, dict) and detailed:
        first_strat = next(iter(detailed))
        rows = detailed[first_strat]
        print(f"Strategy '{first_strat}' count:", len(rows))
        if isinstance(rows, list) and rows:
            print("First item keys:", list(rows[0].keys()))
            print("Sample question field:", {k: rows[0].get(k) for k in ("question_number", "q_num", "question_id", "qid", "id") if k in rows[0]})

def check_metrics(item: dict) -> list[str]:
    """Recompute P@K, R@K, MRR@K from retrieved_chunk_ids and compare with stored metrics."""
    errors = []
    rel_set = set(item.get("relevant_chunk_ids", []))
    ret_ids = item.get("retrieved_chunk_ids", [])
    stored = item.get("metrics", {})

    if not rel_set or not stored:
        return errors

    for k in (1, 3, 5):
        top_k = ret_ids[:k]
        hits = [cid for cid in top_k if cid in rel_set]
        p_k = len(hits) / k
        r_k = len(set(hits)) / len(rel_set)
        mrr_k = 0.0
        for idx, cid in enumerate(top_k, start=1):
            if cid in rel_set:
                mrr_k = 1.0 / idx
                break

        for name, expected in [
            (f"precision@{k}", p_k),
            (f"recall@{k}", r_k),
            (f"mrr@{k}", mrr_k),
        ]:
            if name in stored and abs(stored[name] - expected) > 1e-6:
                errors.append(
                    f"{name}: stored={stored[name]:.6f} vs recomputed={expected:.6f}"
                )
    return errors


def main():
    with INPUT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    detailed = data.get("detailed_results", {})
    strategies = list(detailed.keys())
    print("Strategies found:", strategies)

    # Index by strategy -> question_number
    by_strat = {
    s: {row.get("question_id") or row["question_number"]: row for row in detailed[s]}
    for s in strategies
    }
    all_q = list(dict.fromkeys(qid for s in strategies for qid in by_strat[s]))
    print(f"Total questions: {len(all_q)}\n")

    # 1. Retrieved list length distribution & duplicates per strategy
    print("=== 1. LIST LENGTHS & DUPLICATES ===")
    for s in strategies:
        lengths = Counter()
        dup_questions = []
        for q, row in by_strat[s].items():
            ret = row.get("retrieved_chunk_ids", [])
            lengths[len(ret)] += 1
            if len(ret) != len(set(ret)):
                dup_questions.append(q)
        print(f"  {s}: lengths={dict(lengths)}, questions_with_duplicates={dup_questions[:10]} (total {len(dup_questions)})")

    # 2. Metric calculation verification
    print("\n=== 2. STORED METRIC MATH VERIFICATION ===")
    for s in strategies:
        bad_q = []
        for q, row in by_strat[s].items():
            errs = check_metrics(row)
            if errs:
                bad_q.append((q, errs))
        print(f"  {s}: metric mismatches = {len(bad_q)}")
        for q, errs in bad_q[:5]:
            print(f"    Q{q}: {errs}")

    # 3. ID format / whitespace / child-vs-parent mismatch check
    print("\n=== 3. CHUNK ID FORMAT & PARTIAL MATCH CHECK ===")
    for s in strategies:
        partial_hits = []
        for q, row in by_strat[s].items():
            rel = row.get("relevant_chunk_ids", [])
            ret = row.get("retrieved_chunk_ids", [])
            rel_set = set(rel)
            ret_set = set(ret)
            if not (rel_set & ret_set):
                # Check if same doc prefix exists in retrieved
                rel_docs = {r.split(":")[0].strip() for r in rel}
                ret_docs = {r.split(":")[0].strip() for r in ret}
                if rel_docs & ret_docs:
                    partial_hits.append((q, rel, [r for r in ret if r.split(":")[0].strip() in rel_docs][:3]))
        print(f"  {s}: exact miss BUT same doc_id present = {len(partial_hits)}")
        for q, rel, sample_ret in partial_hits[:5]:
            print(f"    Q{q}: relevant={rel} | retrieved_same_doc={sample_ret}")

    # 4. Hybrid vs Dense/Sparse & Reranker vs Hybrid pool consistency
    print("\n=== 4. FUSION & RERANKER POOL CONSISTENCY ===")
    dense_k = "Dense (Chroma)"
    sparse_k = "Sparse (BM25)"
    hybrid_k = "Hybrid"
    rerank_k = "Hybrid + BGE Reranker"

    if all(k in by_strat for k in (dense_k, sparse_k, hybrid_k, rerank_k)):
        hybrid_outside_union = []
        rerank_outside_hybrid = []
        rerank_pool_sizes = Counter()

        for q in all_q:
            d_ids = by_strat[dense_k][q]["retrieved_chunk_ids"]
            s_ids = by_strat[sparse_k][q]["retrieved_chunk_ids"]
            h_ids = by_strat[hybrid_k][q]["retrieved_chunk_ids"]
            r_ids = by_strat[rerank_k][q]["retrieved_chunk_ids"]

            union_ds = set(d_ids) | set(s_ids)
            extra_in_h = set(h_ids) - union_ds
            if extra_in_h:
                hybrid_outside_union.append((q, len(extra_in_h)))

            extra_in_r = set(r_ids) - set(h_ids)
            if extra_in_r:
                rerank_outside_hybrid.append((q, len(extra_in_r), list(extra_in_r)[:2]))

            # Find deepest hybrid rank used by any reranked item
            h_rank_map = {cid: idx for idx, cid in enumerate(h_ids, start=1)}
            used_ranks = [h_rank_map[cid] for cid in r_ids if cid in h_rank_map]
            if used_ranks:
                rerank_pool_sizes[max(used_ranks)] += 1

        print(f"  Hybrid IDs NOT in (Dense ∪ Sparse): {len(hybrid_outside_union)} questions -> {hybrid_outside_union[:5]}")
        print(f"  Reranker IDs NOT in Hybrid list:    {len(rerank_outside_hybrid)} questions -> {rerank_outside_hybrid[:5]}")
        print(f"  Max Hybrid rank observed inside Reranker output (top 10): {rerank_pool_sizes.most_common(10)}")

        # 5. Deep trace on suspicious questions (q2, q5, q71, q94)
        print("\n=== 5. DETAILED TRACE FOR Q2, Q5, Q71, Q94 ===")
        for q in [2, 5, 71, 94]:
            if q not in by_strat[hybrid_k]:
                continue
            rel = by_strat[hybrid_k][q]["relevant_chunk_ids"]
            print(f"\n  --- Q{q} | relevant={rel} ---")
            for s in strategies:
                ret = by_strat[s][q]["retrieved_chunk_ids"]
                ranks = [idx for idx, cid in enumerate(ret, start=1) if cid in set(rel)]
                print(f"    {s:<22} len={len(ret):<3} rel_ranks={ranks if ranks else 'None'} top3={ret[:3]}")


if __name__ == "__main__":
    #main()
    quick_schema_check()