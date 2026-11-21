from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from src.agents.retriever import adapt_to_retriever_fn, make_retrieval_node
from src.agents.state import AgentRAGState

logger = logging.getLogger(__name__)


def _preview_ids(docs: list[Any], limit: int = 5) -> list[Any]:
    ids: list[Any] = []
    for item in (docs or [])[:limit]:
        if isinstance(item, dict):
            ids.append(item.get("chunk_id") or item.get("doc_id") or item.get("id"))
        else:
            ids.append(
                getattr(item, "chunk_id", None)
                or getattr(item, "doc_id", None)
                or getattr(item, "id", None)
            )
    return ids


def _summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    plan = state.get("plan") or {}
    return {
        "attempt": state.get("attempt", 0),
        "max_attempts": state.get("max_attempts", 2),
        "plan_mode": plan.get("mode") if isinstance(plan, dict) else getattr(plan, "mode", None),
        "queries_count": len(state.get("search_queries") or []),
        "candidates_count": len(state.get("candidates") or []),
        "reranked_ids": _preview_ids(state.get("reranked_docs") or []),
        "evidence_sufficient": state.get("evidence_sufficient", False),
        "gaps_count": len(state.get("evidence_gaps") or []),
        "draft_len": len(state.get("draft_answer") or ""),
        "should_refuse": state.get("should_refuse", False),
        "citations": state.get("citations") or [],
        "verified": state.get("verified", False),
    }


def wrap_node_with_logs(
    node_name: str,
    node_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _wrapped(state: dict[str, Any]) -> dict[str, Any]:
        in_summary = _summarize_state(state)
        logger.info("[ENTER %-17s] %s", node_name, in_summary)
        logger.debug("[ENTER %-17s] queries=%s", node_name, state.get("search_queries") or [])

        updates = node_fn(state) or {}
        merged = {**state, **updates}
        out_summary = _summarize_state(merged)

        logger.info(
            "[EXIT  %-17s] updated_keys=%s | %s",
            node_name,
            list(updates.keys()),
            out_summary,
        )
        if updates.get("verification_notes"):
            logger.debug("[EXIT  %-17s] verification_notes=%s", node_name, updates["verification_notes"])
        if updates.get("evidence_gaps"):
            logger.debug("[EXIT  %-17s] evidence_gaps=%s", node_name, updates["evidence_gaps"])
        return updates

    return _wrapped


def _default_route_from_supervisor(state: dict[str, Any]) -> str:
    if state.get("verified"):
        return "finalize"
    attempt = int(state.get("attempt", 0) or 0)
    max_attempts = int(state.get("max_attempts", 2) or 2)
    if attempt >= max_attempts and (state.get("draft_answer") or state.get("should_refuse")):
        return "finalize"
    return "query_planning"


def _default_route_after_verification(state: dict[str, Any]) -> str:
    if state.get("verified"):
        return "finalize"
    attempt = int(state.get("attempt", 0) or 0)
    max_attempts = int(state.get("max_attempts", 2) or 2)
    if attempt < max_attempts:
        return "supervisor"
    return "finalize"


def _resolve_callable(
    module_path: str,
    names: list[str],
    fallback: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    mod = importlib.import_module(module_path)
    for name in names:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    if fallback is not None:
        return fallback
    raise ImportError(f"None of {names} found in {module_path}")


def _resolve_optional_callable(module_path: str, names: list[str]) -> Callable[..., Any] | None:
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError:
        return None
    for name in names:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    return None


def _default_finalize_node(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_output": {
            "question": state.get("question", ""),
            "answer": state.get("draft_answer", ""),
            "should_refuse": state.get("should_refuse", False),
            "refusal_reason": state.get("refusal_reason", ""),
            "citations": state.get("citations", []),
            "citation_pairs": state.get("citation_pairs", []),
            "reranked_docs": state.get("reranked_docs", []),
            "verified": state.get("verified", False),
            "verification_notes": state.get("verification_notes", ""),
            "attempt": state.get("attempt", 0),
            "trace": state.get("trace", []),
        }
    }


def _build_analyst_node(
    config: Any,
    rerank_fn: Callable[..., Any] | None,
    effective_top_k: int,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    analyst_builder = _resolve_optional_callable(
        "src.agents.analyst",
        ["build_analyst_node", "make_evidence_analysis_node", "make_analyst_node"],
    )
    if analyst_builder is not None and (rerank_fn is not None or effective_top_k != 5 or config is not None):
        for kwargs in (
            {"rerank_fn": rerank_fn, "top_k": effective_top_k, "config": config},
            {"rerank_fn": rerank_fn, "top_k": effective_top_k},
            {"top_k": effective_top_k},
        ):
            filtered = {k: v for k, v in kwargs.items() if v is not None}
            try:
                built = analyst_builder(**filtered)
                if callable(built):
                    return built
            except TypeError:
                continue

    analyst_fn = _resolve_optional_callable(
        "src.agents.analyst",
        ["evidence_analysis_node", "analyst_node", "evidence_analysis"],
    )
    if analyst_fn is not None:
        return analyst_fn
    raise ImportError("Could not resolve an analyst node or builder from src.agents.analyst")


def _build_generator_node(
    config: Any,
    prompt_strategy: Any,
    answer_strategy: Any,
    documents: list[Any] | None,
    model_callable: Callable[[str], dict[str, Any]] | None,
    generate_fn: Callable[..., Any] | None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if callable(generate_fn):
        return generate_fn

    generator_mod = importlib.import_module("src.agents.generator")
    make_gen_node = getattr(generator_mod, "make_answer_generation_node", None)
    make_strat_gen = getattr(generator_mod, "make_answer_strategy_generator", None)

    if answer_strategy is not None and prompt_strategy is not None and callable(make_strat_gen):
        strat_fn = make_strat_gen(
            answer_strategy=answer_strategy,
            prompt_strategy=prompt_strategy,
            documents=documents,
        )
        return make_gen_node(strat_fn) if callable(make_gen_node) else strat_fn

    generator_builder = _resolve_optional_callable(
        "src.agents.generator",
        ["build_generator_node", "make_generator_node"],
    )
    if generator_builder is not None:
        for kwargs in (
            {
                "config": config,
                "prompt_strategy": prompt_strategy,
                "answer_strategy": answer_strategy,
                "model_callable": model_callable,
                "documents": documents,
            },
            {
                "prompt_strategy": prompt_strategy,
                "answer_strategy": answer_strategy,
                "model_callable": model_callable,
                "documents": documents,
            },
        ):
            filtered = {k: v for k, v in kwargs.items() if v is not None}
            try:
                built = generator_builder(**filtered)
                if callable(built):
                    return built
            except TypeError:
                continue

    generator_fn = _resolve_optional_callable(
        "src.agents.generator",
        ["answer_generation_node", "generator_node", "answer_generation"],
    )
    if generator_fn is not None:
        return generator_fn
    raise ImportError("Could not resolve a generator node or builder from src.agents.generator")


def build_agent_rag_graph(
    config: Any = None,
    rag_or_search: Any = None,
    prompt_strategy: Any = None,
    documents: list[Any] | None = None,
    model_callable: Callable[[str], dict[str, Any]] | None = None,
    answer_strategy: Any = None,
    rerank_fn: Callable[..., Any] | None = None,
    search_fn: Callable[..., Any] | None = None,
    top_k: int | None = None,
    generate_fn: Callable[..., Any] | None = None,
):
    """Build and compile the multi-agent RAG StateGraph with trace and routing logs."""
    effective_retriever = rag_or_search if rag_or_search is not None else search_fn
    effective_top_k = top_k if top_k is not None else getattr(config, "agent_top_k", 5)

    logger.info(
        "[GRAPH BUILD] config_search=%s has_search=%s has_generate=%s prompt=%s answer=%s top_k=%s max_attempts=%s",
        getattr(config, "search_strategy", None),
        bool(effective_retriever),
        bool(generate_fn or model_callable or answer_strategy),
        getattr(prompt_strategy, "__class__", type(prompt_strategy)).__name__
        if prompt_strategy is not None
        else getattr(config, "prompt_strategy", None),
        getattr(answer_strategy, "__class__", type(answer_strategy)).__name__
        if answer_strategy is not None
        else getattr(config, "answer_strategy", None),
        effective_top_k,
        getattr(config, "agent_max_attempts", 2),
    )

    supervisor_fn = _resolve_callable(
        "src.agents.supervisor",
        ["supervisor_node", "supervisor"],
    )
    supervisor_router = _resolve_callable(
        "src.agents.supervisor",
        ["route_from_supervisor", "supervisor_route"],
        fallback=_default_route_from_supervisor,
    )
    planner_fn = _resolve_callable(
        "src.agents.planner",
        ["query_planning_node", "planner_node", "query_planning"],
    )
    retriever_builder = _resolve_optional_callable(
        "src.agents.retriever",
        ["build_retriever_node", "make_retriever_node", "build_retriever"],
    )
    retriever_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None

    verifier_fn = _resolve_callable(
        "src.agents.verifier",
        ["verification_node", "verifier_node", "verification"],
    )
    verifier_router = _resolve_callable(
        "src.agents.verifier",
        ["route_after_verification", "verifier_route"],
        fallback=_default_route_after_verification,
    )
    finalize_fn = (
        _resolve_optional_callable("src.agents.finalize", ["finalize_node", "finalize"])
        or _resolve_optional_callable("src.agents.finalizer", ["finalize_node", "finalizer_node", "finalize"])
        or _resolve_optional_callable("src.agents.nodes", ["finalize_node", "finalize"])
        or _default_finalize_node
    )

    if retriever_builder is not None:
        for kwargs in (
            {
                "config": config,
                "rag_or_search": effective_retriever,
                "search_fn": effective_retriever,
                "documents": documents,
                "top_k": effective_top_k,
            },
            {
                "config": config,
                "rag_or_search": effective_retriever,
                "documents": documents,
            },
        ):
            filtered = {k: v for k, v in kwargs.items() if v is not None}
            try:
                built = retriever_builder(**filtered)
                if callable(built):
                    retriever_fn = built
                    break
            except TypeError:
                continue
        if retriever_fn is None:
            try:
                retriever_fn = retriever_builder(effective_retriever, documents=documents)
            except TypeError:
                retriever_fn = None

    if retriever_fn is None:
        adapted_search = adapt_to_retriever_fn(
            effective_retriever,
            documents=documents,
            limit=max(effective_top_k * 2, 8),
        )
        retriever_fn = make_retrieval_node(adapted_search)

    analyst_fn = _build_analyst_node(
        config=config,
        rerank_fn=rerank_fn,
        effective_top_k=effective_top_k,
    )
    generator_fn = _build_generator_node(
        config=config,
        prompt_strategy=prompt_strategy,
        answer_strategy=answer_strategy,
        documents=documents,
        model_callable=model_callable,
        generate_fn=generate_fn,
    )

    def _logged_supervisor_route(state: dict[str, Any]) -> str:
        dest = supervisor_router(state)
        logger.info(
            "[ROUTE supervisor -> %-14s] attempt=%s/%s verified=%s",
            dest,
            state.get("attempt", 0),
            state.get("max_attempts", 2),
            state.get("verified", False),
        )
        return dest

    def _logged_verifier_route(state: dict[str, Any]) -> str:
        dest = verifier_router(state)
        logger.info(
            "[ROUTE verifier   -> %-14s] verified=%s sufficient=%s attempt=%s/%s gaps=%s",
            dest,
            state.get("verified", False),
            state.get("evidence_sufficient", False),
            state.get("attempt", 0),
            state.get("max_attempts", 2),
            state.get("evidence_gaps") or [],
        )
        return dest

    workflow = StateGraph(AgentRAGState)

    workflow.add_node("supervisor", wrap_node_with_logs("supervisor", supervisor_fn))
    workflow.add_node("query_planning", wrap_node_with_logs("query_planning", planner_fn))
    workflow.add_node("retrieval", wrap_node_with_logs("retrieval", retriever_fn))
    workflow.add_node("evidence_analysis", wrap_node_with_logs("evidence_analysis", analyst_fn))
    workflow.add_node("answer_generation", wrap_node_with_logs("answer_generation", generator_fn))
    workflow.add_node("verification", wrap_node_with_logs("verification", verifier_fn))
    workflow.add_node("finalize", wrap_node_with_logs("finalize", finalize_fn))

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        _logged_supervisor_route,
        {
            "query_planning": "query_planning",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("query_planning", "retrieval")
    workflow.add_edge("retrieval", "evidence_analysis")
    workflow.add_edge("evidence_analysis", "answer_generation")
    workflow.add_edge("answer_generation", "verification")
    workflow.add_conditional_edges(
        "verification",
        _logged_verifier_route,
        {
            "supervisor": "supervisor",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("finalize", END)

    return workflow.compile()


# Alias for callers expecting build_rag_graph
build_rag_graph = build_agent_rag_graph