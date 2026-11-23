"""Small Streamlit UI for the injectable financial-document pipeline."""

import inspect
import json
import logging
from pathlib import Path
import time
from typing import Any
import uuid

import streamlit as st

from src.config import AppConfig
from src.data import load_saved_sample
from src.evaluation.ragas_eval import run_ragas_evaluation
from src.logging_config import configure_logging
from src.models.records import Document, QAPair
from src.pipeline import build_strategies_from_config, run_pipeline
from src.agents.graph import format_query_flow

configure_logging()

logger = logging.getLogger(__name__)


def _maybe_index_documents(strategies: Any, documents: list[Document], config: AppConfig) -> None:
    if config.search_strategy not in {"chroma", "hybrid", "hybrid_reranker", "hybrid-reranker"}:
        return
    if not documents:
        return

    search = getattr(getattr(strategies, "rag", None), "text", None)
    search = getattr(search, "search", None)
    if search is None:
        return

    index_fn = getattr(search, "index_documents", None)
    if callable(index_fn):
        index_fn(documents)


def _sanitize_chunk_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    # Guard against accidental shortcode expansion of ":child:"
    return text.replace("🧒", ":child:")


def _load_documents_from_500_or_default() -> tuple[list, str]:
    sample_500_path = Path("data/sample_500/documents.json")
    print(f'sample_500_path: {sample_500_path} :: {sample_500_path.exists()}')
    if sample_500_path.exists():
        raw_docs = json.loads(sample_500_path.read_text(encoding="utf-8"))
        documents = [
            Document(
                doc_id=entry["doc_id"],
                content=entry.get("content", ""),
                qa_pairs=[
                    QAPair(
                        question=qa.get("question", ""),
                        answer=qa.get("answer", ""),
                    )
                    for qa in entry.get("qa_pairs", [])
                ],
                image_path=(
                    entry["image_path"].replace("\\", "/")
                    if entry.get("image_path")
                    else None
                ),
            )
            for entry in raw_docs
        ]
        print(f"len: {len(documents)}")
        return documents, str(sample_500_path)

    return load_saved_sample(), "default_sample"


@st.cache_resource
def load_app():
    logger.info("APPLICATION_INITIALIZATION_STARTED")
    config = AppConfig.from_env()
    logger.info(
        "CONFIG_LOADED | search=%s | rag=%s | answer=%s | prompt=%s",
        config.search_strategy,
        config.rag_strategy,
        config.answer_strategy,
        config.prompt_strategy,
    )
    documents, dataset_source = _load_documents_from_500_or_default()

    model_callable = None

    if config.answer_strategy == "llm":
        from src.generate.generate_answer import build_model_callable

        logger.info("LLM_INITIALIZATION_STARTED")
        model_callable = build_model_callable()
        logger.info("LLM_INITIALIZATION_COMPLETED")

    strategies = build_strategies_from_config(
        config,
        model_callable=model_callable,
    )
    # _maybe_index_documents(strategies, documents, config)

    logger.info(
        "PIPELINE_INITIALIZED | source=%s | documents=%d | answerer=%s | rag=%s",
        dataset_source,
        len(documents),
        type(strategies.answerer).__name__,
        type(strategies.rag).__name__,
    )

    logger.info("APPLICATION_INITIALIZATION_COMPLETED")

    return documents, strategies, config, dataset_source


def _get_rerank_summary(item) -> tuple[int | float | None, int | None]:
    dense_r = getattr(item, "dense_rank", None)
    sparse_r = getattr(item, "sparse_rank", None)
    hybrid_r = getattr(item, "hybrid_rank", None)
    rerank_r = getattr(item, "rerank_rank", None)

    pre_r = hybrid_r
    if pre_r is None:
        candidates = [r for r in (dense_r, sparse_r) if r is not None]
        pre_r = min(candidates) if candidates else None

    if pre_r is None:
        pre_r = getattr(item, "pre_rerank_rank", None)

    if pre_r is None or rerank_r is None:
        return pre_r, getattr(item, "rank_gain", None)
    return pre_r, int(pre_r) - int(rerank_r)


def _format_citations(citations_list: list) -> list:
    formatted = []
    for entry in citations_list:
        if isinstance(entry, dict):
            cleaned = dict(entry)
            if "chunk_id" in cleaned:
                cleaned["chunk_id"] = _sanitize_chunk_id(cleaned["chunk_id"])
            formatted.append(cleaned)
        else:
            formatted.append(_sanitize_chunk_id(entry))
    return formatted


def _extract_cited_keys(citations_list: list) -> tuple[set[str], set[str]]:
    cited_doc_ids: set[str] = set()
    cited_chunk_ids: set[str] = set()
    for entry in citations_list:
        if isinstance(entry, dict):
            doc_id = entry.get("doc_id")
            if doc_id:
                cited_doc_ids.add(str(doc_id).strip())
            cid = _sanitize_chunk_id(entry.get("chunk_id"))
            if cid:
                cited_chunk_ids.add(cid)
    return cited_doc_ids, cited_chunk_ids


def _is_cited_item(item: Any, cited_doc_ids: set[str], cited_chunk_ids: set[str]) -> bool:
    cid = _sanitize_chunk_id(getattr(item, "chunk_id", None))
    doc_id = str(getattr(item, "doc_id", "") or "").strip()
    if cid and cid in cited_chunk_ids:
        return True
    if not cited_chunk_ids and doc_id and doc_id in cited_doc_ids:
        return True
    return False


def _extract_rewritten_query(payload: Any, original_question: str) -> str | None:
    rewritten = ""
    if isinstance(payload, dict):
        rewritten = str(
            payload.get("rewritten_query")
            or payload.get("planned_query")
            or payload.get("query")
            or ""
        ).strip()
    elif isinstance(payload, str):
        rewritten = payload.strip()

    if rewritten and rewritten != original_question.strip():
        return rewritten
    return None


def _invoke_pipeline_with_stages(
    *,
    question: str,
    documents: list,
    session_id: str,
    image: Any,
    strategies: Any,
    active_banner,
    trace_box,
    step_delay_s: float = 0.14,
) -> tuple[dict, list[str]]:
    trace_lines: list[str] = []

    def _push_stage(active_msg: str, done_msg: str) -> None:
        active_banner.info(active_msg)
        if done_msg not in trace_lines:
            trace_lines.append(done_msg)
            trace_box.markdown("\n".join(f"- {line}" for line in trace_lines))
        if step_delay_s > 0:
            time.sleep(step_delay_s)

    def _on_stage(stage: Any, payload: Any = None) -> None:
        stage_name = str(stage or "").lower().strip()
        if stage_name == "supervisor":
            _push_stage(
                "🧭 **Supervisor Agent working...** Routing query across pipeline stages.",
                "🧭 **Supervisor Agent:** Routed query to pipeline.",
            )
        elif stage_name == "planner":
            rewritten = _extract_rewritten_query(payload, question)
            if rewritten:
                _push_stage(
                    f"📝 **Planner Agent rewriting query...** → `{rewritten}`",
                    f"📝 **Planner Agent:** Rewrote query → `{rewritten}`",
                )
            else:
                _push_stage(
                    "📝 **Planner Agent working...** Evaluating query & search plan.",
                    "📝 **Planner Agent:** Checked query plan.",
                )
        elif stage_name in {"retriever", "retrieval"}:
            count_note = ""
            if isinstance(payload, int):
                count_note = f" ({payload} chunks)"
            elif isinstance(payload, dict) and payload.get("count") is not None:
                count_note = f" ({payload['count']} chunks)"
            _push_stage(
                f"🔎 **Retriever Agent working...** Searching & reranking evidence{count_note}.",
                f"🔎 **Retriever Agent:** Retrieved evidence{count_note}.",
            )
        elif stage_name in {"llm", "generator", "analyst"}:
            _push_stage(
                "🤖 **Final LLM working...** Synthesizing grounded answer from evidence.",
                "🤖 **Final LLM:** Generated grounded response.",
            )
        elif stage_name == "verifier":
            _push_stage(
                "✅ **Verifier Agent working...** Checking citations & grounding.",
                "✅ **Verifier Agent:** Verified response & citations.",
            )

    sig = inspect.signature(run_pipeline)
    kwargs: dict[str, Any] = {
        "session_id": session_id,
        "image": image,
        "strategies": strategies,
    }
    if "on_stage" in sig.parameters:
        kwargs["on_stage"] = _on_stage

    result = run_pipeline(question, documents, **kwargs)
    return result, trace_lines


def _replay_completed_stages(
    *,
    result: dict[str, Any],
    question: str,
    active_banner,
    trace_box,
    existing_lines: list[str],
    step_delay_s: float = 0.14,
) -> list[str]:
    lines = list(existing_lines)
    steps = [str(s).lower().strip() for s in (result.get("steps") or []) if s]
    evidence_list = result.get("evidence") or []
    answer_obj = result.get("answer") or {}
    citations_list = _format_citations(answer_obj.get("citations") or [])
    rewritten_query = (
        result.get("rewritten_query")
        or result.get("planned_query")
        or answer_obj.get("rewritten_query")
        or (result.get("metrics") or {}).get("rewritten_query")
    )

    stage_sequence = steps or ["supervisor", "planner", "retriever", "llm", "verifier"]

    for stg in stage_sequence:
        if any(tag in stg for tag in ("super", "route", "start")):
            active_msg = "🧭 **Supervisor Agent working...** Routing query across pipeline stages."
            done_msg = f"🧭 **Supervisor Agent:** Routed query ({len(evidence_list)} chunks retrieved)."
        elif "plan" in stg or "rewrit" in stg:
            rewritten = _extract_rewritten_query(rewritten_query, question)
            if rewritten:
                active_msg = f"📝 **Planner Agent rewriting query...** → `{rewritten}`"
                done_msg = f"📝 **Planner Agent:** Rewrote query → `{rewritten}`"
            else:
                active_msg = "📝 **Planner Agent working...** Checking query & retrieval plan."
                done_msg = "📝 **Planner Agent:** Checked query plan (no rewrite needed)."
        elif "retriev" in stg or "search" in stg or "rerank" in stg:
            active_msg = f"🔎 **Retriever Agent working...** Searching & reranking ({len(evidence_list)} chunks)."
            done_msg = f"🔎 **Retriever Agent:** Retrieved {len(evidence_list)} evidence chunk(s)."
        elif any(tag in stg for tag in ("llm", "generat", "answer", "analy")):
            active_msg = "🤖 **Final LLM working...** Synthesizing grounded response."
            done_msg = "🤖 **Final LLM:** Generated grounded answer."
        elif "verif" in stg or "cit" in stg:
            active_msg = "✅ **Verifier Agent working...** Checking citations & grounding."
            if answer_obj.get("should_refuse"):
                done_msg = (
                    f"✅ **Verifier Agent:** Refusal verified "
                    f"({answer_obj.get('reason') or 'insufficient evidence'})."
                )
            elif citations_list:
                done_msg = f"✅ **Verifier Agent:** Verified {len(citations_list)} citation(s)."
            else:
                done_msg = "⚠️ **Verifier Agent:** Completed with no citations."
        else:
            active_msg = f"⚙️ **{stg} working...**"
            done_msg = f"⚙️ **{stg}:** Completed."

        active_banner.info(active_msg)
        if done_msg not in lines:
            lines.append(done_msg)
            trace_box.markdown("\n".join(f"- {line}" for line in lines))
        if step_delay_s > 0:
            time.sleep(step_delay_s)

    return lines


def _render_latest_result(result: dict[str, Any]) -> None:
    answer_obj = result.get("answer", {})
    evidence_list = result.get("evidence", [])
    citations_list = _format_citations(answer_obj.get("citations") or [])
    cited_doc_ids, cited_chunk_ids = _extract_cited_keys(citations_list)
    agent_trace = result.get("_agent_trace") or []

    boosted_items = []
    for item in evidence_list:
        pre_r, gain = _get_rerank_summary(item)
        if gain is not None and gain > 0:
            boosted_items.append((item, pre_r, gain))

    if cited_chunk_ids:
        relevant_boosted = [
            entry
            for entry in boosted_items
            if _sanitize_chunk_id(getattr(entry[0], "chunk_id", None)) in cited_chunk_ids
        ]
    elif cited_doc_ids:
        relevant_boosted = [
            entry
            for entry in boosted_items
            if str(getattr(entry[0], "doc_id", "") or "").strip() in cited_doc_ids
        ]
    else:
        relevant_boosted = boosted_items

    st.subheader("Final Answer")
    if not evidence_list:
        st.warning("No evidence chunks were retrieved for this query.")
    elif relevant_boosted:
        st.success(
            f"Reranker promoted {len(relevant_boosted)} cited chunk(s) "
            f"used for answer grounding."
        )

    if answer_obj.get("should_refuse"):
        st.info(f"Refusal: {answer_obj.get('reason') or 'Insufficient grounded evidence.'}")

    st.write(answer_obj.get("answer", ""))
    st.caption(f"Citations: {citations_list or 'None'}")

    if relevant_boosted:
        st.markdown("**Reranker Promotion Confirmations (Cited Evidence)**")
        for item, pre_r, gain in relevant_boosted:
            cid = _sanitize_chunk_id(getattr(item, "chunk_id", None)) or "Not available"
            rerank_r = getattr(item, "rerank_rank", None)
            st.caption(
                f"- `{item.doc_id}` | Chunk `{cid}`: "
                f"#{int(pre_r)} → #{int(rerank_r)} (**+{gain}**)"
            )

    cited_first = [
        item for item in evidence_list if _is_cited_item(item, cited_doc_ids, cited_chunk_ids)
    ]
    other_items = [
        item for item in evidence_list if not _is_cited_item(item, cited_doc_ids, cited_chunk_ids)
    ]
    displayed_evidence = cited_first + other_items

    st.markdown("**Graph Flow for This Query**")
    st.code(format_query_flow(result), language="text")
    with st.expander("Pipeline details"):
        if agent_trace:
            st.markdown("**Agent execution trace**")
            st.markdown("\n".join(f"- {line}" for line in agent_trace))
        st.write("Conversation history", result.get("history", []))
        st.write(" → ".join(result.get("steps", [])))
        st.json(result.get("metrics", {}))
        st.write(f"Retrieved evidence ({len(displayed_evidence)} shown)")

        for index, item in enumerate(displayed_evidence, start=1):
            chunk_id = _sanitize_chunk_id(getattr(item, "chunk_id", None)) or "Not available"
            dense_r = getattr(item, "dense_rank", None)
            sparse_r = getattr(item, "sparse_rank", None)
            hybrid_r = getattr(item, "hybrid_rank", None)
            rerank_r = getattr(item, "rerank_rank", None)
            _, gain = _get_rerank_summary(item)
            score_val = getattr(item, "score", None)

            cited_badge = (
                " | **Cited**"
                if _is_cited_item(item, cited_doc_ids, cited_chunk_ids)
                else ""
            )
            boost_badge = f" | Reranker Boosted (+{gain})" if (gain is not None and gain > 0) else ""
            score_str = f"{float(score_val):.4f}" if score_val is not None else "N/A"
            st.markdown(
                f"**{index}. `{item.doc_id}`** | Chunk ID: `{chunk_id}`{cited_badge}{boost_badge}"
            )
            st.caption(
                f"Relevance score: {score_str} | dense={dense_r}, sparse={sparse_r}, "
                f"hybrid={hybrid_r}, rerank={rerank_r}"
            )

            chunk_text = (
                getattr(item, "text", None)
                or getattr(item, "content", None)
                or getattr(item, "page_content", None)
                or ""
            )
            char_count = len(chunk_text)
            st.write(f"Chunk character count: `{char_count}`")
            with st.expander("View retrieved chunk text"):
                st.code(chunk_text)


def _render_ragas_section() -> None:
    st.divider()
    st.subheader("RAGAS Evaluation")
    st.caption(f"Captured queries ready for evaluation: {len(st.session_state.eval_records)}")

    col_run, col_clear = st.columns([1, 1])
    with col_run:
        run_eval_clicked = st.button(
            "Run RAGAS Evaluation",
            disabled=not st.session_state.eval_records,
        )
    with col_clear:
        if st.button("Clear Captured Runs", disabled=not st.session_state.eval_records):
            st.session_state.eval_records = []
            st.session_state.ragas_summary = None
            st.session_state.ragas_rows = []
            st.rerun()

    if run_eval_clicked:
        try:
            with st.spinner("Computing RAGAS scores across captured queries..."):
                summary, rows = run_ragas_evaluation(st.session_state.eval_records)
            st.session_state.ragas_summary = summary
            st.session_state.ragas_rows = rows
        except Exception as exc:
            logger.exception("RAGAS_EVALUATION_FAILED")
            st.error(f"RAGAS evaluation failed: {exc}")

    if st.session_state.ragas_summary:
        metric_cols = st.columns(len(st.session_state.ragas_summary))
        for m_col, (metric_name, score) in zip(
            metric_cols, st.session_state.ragas_summary.items()
        ):
            m_col.metric(label=metric_name, value=f"{score:.4f}")

    if st.session_state.ragas_rows:
        st.dataframe(st.session_state.ragas_rows, width="stretch")


def main():
    st.set_page_config(page_title="Financial Document Copilot", page_icon="📄")
    st.title("Financial Document Copilot")
    st.caption("Answers are retrieved from indexed financial documents.")

    try:
        documents, strategies, config, dataset_source = load_app()
    except FileNotFoundError:
        st.error("Sample data is missing. Run `uv run python -m scripts.download_sample`.")
        return
    except ValueError as exc:
        st.error(f"Invalid strategy config: {exc}")
        return

    st.write(f"Loaded {len(documents)} documents (`{dataset_source}`).")
    st.caption(
        f"Config: search={config.search_strategy}, rag={config.rag_strategy}, "
        f"prompt={config.prompt_strategy}, answer={config.answer_strategy}"
    )

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "eval_records" not in st.session_state:
        st.session_state.eval_records = []
    if "ragas_summary" not in st.session_state:
        st.session_state.ragas_summary = None
    if "ragas_rows" not in st.session_state:
        st.session_state.ragas_rows = []

    question = st.text_input("Ask a question about the documents")
    uploaded_image = st.file_uploader("Optional image", type=["png", "jpg", "jpeg"])

    if st.button("Ask") and question:
        image = uploaded_image.getvalue() if uploaded_image else None
        logger.info(
            "USER_QUERY_SUBMITTED | question_length=%d | session_id=%s",
            len(question),
            st.session_state.session_id,
        )
        try:
            with st.status("Multi-agent pipeline running...", expanded=True) as status_box:
                active_banner = st.empty()
                trace_box = st.empty()
                active_banner.info("🧭 **Supervisor Agent working...** Starting pipeline.")

                result, trace_lines = _invoke_pipeline_with_stages(
                    question=question,
                    documents=documents,
                    session_id=st.session_state.session_id,
                    image=image,
                    strategies=strategies,
                    active_banner=active_banner,
                    trace_box=trace_box,
                    step_delay_s=0.14,
                )

                trace_lines = _replay_completed_stages(
                    result=result,
                    question=question,
                    active_banner=active_banner,
                    trace_box=trace_box,
                    existing_lines=trace_lines,
                    step_delay_s=0.14,
                )

                result["_agent_trace"] = trace_lines
                active_banner.success("✨ **Pipeline finished** — answer ready below.")
                trace_box.markdown("\n".join(f"- {line}" for line in trace_lines))

                status_box.update(
                    label="Multi-agent pipeline completed",
                    state="complete",
                    expanded=False,
                )
            logger.info("UI_QUERY_COMPLETED")
        except ValueError as exc:
            st.warning(str(exc))
            return
        except Exception:
            logger.exception("UI_QUERY_FAILED")
            st.error("The query failed. Check logs/application.log for details.")
            return

        st.session_state.last_result = result

        answer_obj = result.get("answer", {})
        evidence_list = result.get("evidence", [])
        st.session_state.eval_records.append(
            {
                "question": question,
                "answer": answer_obj.get("answer", ""),
                "evidence": evidence_list,
                "should_refuse": bool(answer_obj.get("should_refuse", False)),
                "reason": str(answer_obj.get("reason") or ""),
            }
        )

    if st.session_state.last_result is not None:
        _render_latest_result(st.session_state.last_result)

    _render_ragas_section()


if __name__ == "__main__":
    main()