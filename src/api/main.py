"""FastAPI service exposing pipeline query and metrics dashboard APIs."""

from pathlib import Path
from time import perf_counter
from contextlib import asynccontextmanager
from collections import Counter
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.config import AppConfig
from src.data import load_dataset_subset, load_saved_sample, save_sample
from src.db import MetricsStore
from src.pipeline import build_strategies_from_config, run_pipeline
from src.pipeline.factory import build_agent_rag_graph

from .schemas import (
    EvaluationRequest,
    EvaluationResponse,
    MetricsCreateRequest,
    MetricsListResponse,
    QueryRequest,
    QueryResponse,
    RagSetupRequest,
    RetrievalReportRequest,
    RetrievalTraceRequest,
    TraceAnalysisRequest,
)

import logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    startup()
    yield


app = FastAPI(title="Financial Copilot API", version="0.1.0", lifespan=lifespan)

_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_DIR = _ROOT / "frontend"
_SAMPLE_IMAGE_DIR = _ROOT / "data" / "sample" / "images"

app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")
app.mount("/sample-images", StaticFiles(directory=str(_SAMPLE_IMAGE_DIR)), name="sample_images")


def _strategy_snapshot(config: AppConfig) -> dict:
    return {
        "search_strategy": config.search_strategy,
        "rag_strategy": config.rag_strategy,
        "prompt_strategy": config.prompt_strategy,
        "answer_strategy": config.answer_strategy,
        "memory_strategy": config.memory_strategy,
        "evaluator_strategy": config.evaluator_strategy,
        "chunking_strategy": config.chunking_strategy,
        "embedder_strategy": config.embedder_strategy,
    }


def _ensure_state() -> None:
    required = ("config", "strategies", "metrics_store", "documents")
    if all(hasattr(app.state, name) for name in required):
        return
    startup()


def _item_field(item, key: str, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _maybe_index_documents(config: AppConfig, strategies, documents) -> None:
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


def _to_citation_pairs(final_output: dict, reranked_docs: list[dict]) -> list[dict]:
    pairs = final_output.get("citation_pairs") or []
    if pairs:
        return [
            {
                "doc_id": str(item.get("doc_id", "")),
                "chunk_id": str(item.get("chunk_id", "")),
            }
            for item in pairs
            if isinstance(item, dict) and item.get("doc_id")
        ]
    by_chunk = {
        str(item.get("chunk_id")): str(item.get("doc_id"))
        for item in reranked_docs
        if item.get("chunk_id") and item.get("doc_id")
    }
    citations = []
    for chunk_id in final_output.get("citations", []) or []:
        doc_id = by_chunk.get(str(chunk_id))
        if doc_id:
            citations.append({"doc_id": doc_id, "chunk_id": str(chunk_id)})
    return citations


def _run_agent_query(question: str, session_id: str, expected_answer: str | None = None) -> dict:
    config = app.state.config
    graph = build_agent_rag_graph(
        config=config,
        rag_or_search=app.state.strategies.rag,
        prompt_strategy=app.state.strategies.prompt,
        documents=app.state.documents,
        model_callable=getattr(app.state, "model_callable", None),
    )
    initial_state = {
        "question": question,
        "attachment": None,
        "plan": {"mode": "initial_search", "focus_gaps": [], "feedback": ""},
        "search_queries": [],
        "doc_scope": None,
        "candidates": [],
        "reranked_docs": [],
        "evidence_sufficient": False,
        "evidence_gaps": [],
        "draft_answer": "",
        "should_refuse": False,
        "refusal_reason": "",
        "citations": [],
        "citation_pairs": [],
        "verified": False,
        "verification_notes": "",
        "attempt": 0,
        "max_attempts": config.agent_max_attempts,
        "trace": [],
        "final_output": {},
    }
    state = graph.invoke(initial_state)
    final_output = state.get("final_output", {})
    reranked_docs = state.get("reranked_docs", []) or []
    citations = _to_citation_pairs(final_output, reranked_docs)
    answer = {
        "answer": final_output.get("answer", ""),
        "citations": citations,
        "should_refuse": bool(final_output.get("should_refuse", state.get("should_refuse", False))),
        "reason": final_output.get("refusal_reason", state.get("refusal_reason", "")),
    }
    metrics = {
        "retrieved_count": len(reranked_docs),
        "has_citation": bool(citations),
        "refused": bool(answer["should_refuse"]),
        "verified": bool(final_output.get("verified", state.get("verified", False))),
    }
    if expected_answer is not None:
        metrics["exact_match"] = float(answer["answer"].strip().casefold() == expected_answer.strip().casefold())
    return {
        "answer": answer,
        "route": "text",
        "evidence": reranked_docs,
        "history": app.state.strategies.memory.get(session_id),
        "metrics": metrics,
        "steps": list(state.get("trace", [])) or ["agent_graph_execution"],
        "_agent_trace": list(state.get("trace", [])),
    }


def startup():
    config = AppConfig.from_env()
    logger.info(
        "APP_STARTUP_CONFIG | search=%s | embedder=%s | model=%s | collection=%s",
        config.search_strategy,
        config.embedder_strategy,
        config.embedding_model_name,
        config.chroma_collection,
    )
    app.state.config = config
    model_callable = None
    if config.answer_strategy == "llm":
        from src.generate.generate_answer import build_model_callable

        model_callable = build_model_callable()
    app.state.model_callable = model_callable

    app.state.strategies = build_strategies_from_config(
        config,
        model_callable=model_callable,
    )
    app.state.metrics_store = MetricsStore(config.metrics_db_path)

    try:
        app.state.documents = load_saved_sample()
        _maybe_index_documents(config, app.state.strategies, app.state.documents)

    except FileNotFoundError as exc:
        raise RuntimeError(
            "Missing data/sample/documents.json. "
            "Run `uv run python -m scripts.download_sample`."
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize document retrieval index. "
            "Check ChromaDB configuration, embedding model and collection."
        ) from exc


def _compute_rank_gain(
    dense_rank: int | float | None,
    sparse_rank: int | float | None,
    hybrid_rank: int | None,
    rerank_rank: int | None,
) -> tuple[int | float | None, int | None]:
    pre_rerank = hybrid_rank
    if pre_rerank is None:
        candidates = [r for r in (dense_rank, sparse_rank) if r is not None]
        pre_rerank = min(candidates) if candidates else None

    if pre_rerank is None or rerank_rank is None:
        return pre_rerank, None
    return pre_rerank, int(pre_rerank) - int(rerank_rank)


def _serialize_evidence(evidence_items) -> list[dict]:
    serialized = []
    for item in evidence_items:
        dense_rank = _item_field(item, "dense_rank", None)
        sparse_rank = _item_field(item, "sparse_rank", None)
        hybrid_rank = _item_field(item, "hybrid_rank", None)
        rerank_rank = _item_field(item, "rerank_rank", None)

        pre_rerank_rank, rank_gain = _compute_rank_gain(
            dense_rank=dense_rank,
            sparse_rank=sparse_rank,
            hybrid_rank=hybrid_rank,
            rerank_rank=rerank_rank,
        )
        if rank_gain is None and _item_field(item, "rank_gain", None) is not None:
            rank_gain = _item_field(item, "rank_gain")

        serialized.append(
            {
                "doc_id": _item_field(item, "doc_id", ""),
                "chunk_id": _item_field(item, "chunk_id", None),
                "question": _item_field(item, "question", None),
                "answer": _item_field(item, "answer", None),
                "chunk_text": (_item_field(item, "text", "") or "")[:4000],
                "image_path": _item_field(item, "image_path", None),
                "score": round(float(_item_field(item, "score", 0.0) or 0.0), 4),
                "dense_rank": dense_rank,
                "sparse_rank": sparse_rank,
                "hybrid_rank": hybrid_rank,
                "rerank_rank": rerank_rank,
                "pre_rerank_rank": pre_rerank_rank,
                "rank_gain": rank_gain,
            }
        )
    return serialized


def _critiques(answer: dict, evidence: list[dict]) -> list[str]:
    critiques = []
    if answer.get("should_refuse"):
        critiques.append(f"Refusal: {answer.get('reason', 'No reason provided')}")
    if not evidence:
        critiques.append("No evidence chunks retrieved.")
    if not answer.get("citations"):
        critiques.append("No citations were attached.")
    else:
        critiques.append("Citations were attached for retrieved evidence.")
    if evidence and all(item.get("score", 0) < 0.3 for item in evidence):
        critiques.append("Similarity scores are low; answer confidence should be treated carefully.")
    boosted = [item for item in evidence if (item.get("rank_gain") or 0) > 0]
    if boosted:
        critiques.append(
            f"Reranker promoted {len(boosted)} retrieved chunk(s) prior to answer generation."
        )
    return critiques


def _failure_category(
    expected_answer: str,
    predicted_answer: str,
    should_refuse: bool,
    citations: list[dict],
    expected_doc_id: str,
    retrieved_count: int,
) -> str:
    if retrieved_count == 0:
        return "retrieval_miss"
    if citations and not any(item.get("doc_id") == expected_doc_id for item in citations):
        return "citation_error"
    if should_refuse:
        return "reasoning_error"
    if predicted_answer.strip().casefold() != expected_answer.strip().casefold():
        return "hallucination"
    return "none"


def _image_url(image_path: str | None) -> str:
    if not image_path:
        return ""
    return f"/sample-images/{Path(image_path).name}"


@app.get("/")
def home():
    return FileResponse(str(_FRONTEND_DIR / "pages.html"))


@app.get("/query")
def query_page():
    return FileResponse(str(_FRONTEND_DIR / "index.html"))


@app.get("/pages")
def pages():
    return FileResponse(str(_FRONTEND_DIR / "pages.html"))


@app.get("/rag")
def rag_setup_page():
    return FileResponse(str(_FRONTEND_DIR / "rag.html"))


@app.get("/dashboard")
def dashboard():
    return FileResponse(str(_FRONTEND_DIR / "traces.html"))


@app.get("/evaluation")
def evaluation_page():
    return FileResponse(str(_FRONTEND_DIR / "evaluation.html"))


@app.get("/api/health")
def health():
    _ensure_state()
    return {"status": "ok", "documents": len(app.state.documents)}


@app.get("/api/config")
def config():
    _ensure_state()
    return _strategy_snapshot(app.state.config)


@app.post(
    "/api/query",
    response_model=QueryResponse,
    responses={
        400: {
            "description": "Invalid query scope, missing document ID, or unknown document ID.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "doc_id is required for document scope"
                    }
                }
            },
        }
    },
)
def query(payload: QueryRequest):
    _ensure_state()
    started = perf_counter()
    use_agent_graph = (
        payload.use_agent_graph
        if payload.use_agent_graph is not None
        else bool(app.state.config.use_multi_agent_graph)
    )
    try:
        if use_agent_graph:
            result = _run_agent_query(
                question=payload.question,
                session_id=payload.session_id,
            )
        else:
            result = run_pipeline(
                question=payload.question,
                documents=app.state.documents,
                session_id=payload.session_id,
                strategies=app.state.strategies,
                scope=payload.scope,
                doc_id=payload.doc_id,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    latency_ms = (perf_counter() - started) * 1000
    metrics = result["metrics"]
    serialized_evidence = _serialize_evidence(result["evidence"])
    trace_evidence = [
        {
            "doc_id": item["doc_id"],
            "chunk_id": item["chunk_id"],
            "question": item["question"],
            "answer": item["answer"],
            "chunk_text": item["chunk_text"],
            "image_path": item["image_path"],
            "score": item["score"],
            "dense_rank": item["dense_rank"],
            "sparse_rank": item["sparse_rank"],
            "hybrid_rank": item["hybrid_rank"],
            "rerank_rank": item["rerank_rank"],
            "rank_gain": item["rank_gain"],
        }
        for item in serialized_evidence
    ]

    app.state.metrics_store.insert_run(
        {
            "run_name": payload.run_name,
            "question": payload.question,
            "answer": result["answer"]["answer"],
            "route": result["route"],
            **_strategy_snapshot(app.state.config),
            "retrieved_count": metrics.get("retrieved_count", len(serialized_evidence)),
            "has_citation": metrics.get("has_citation", bool(result["answer"].get("citations"))),
            "refused": metrics.get("refused", bool(result["answer"].get("should_refuse"))),
            "exact_match": metrics.get("exact_match"),
            "latency_ms": latency_ms,
        }
    )
    app.state.metrics_store.insert_query_trace(
        {
            "run_name": payload.run_name,
            "session_id": payload.session_id,
            "question": payload.question,
            "answer": result["answer"]["answer"],
            "route": result["route"],
            "latency_ms": latency_ms,
            "strategies": _strategy_snapshot(app.state.config),
            "metrics": {**metrics, "latency_ms": round(latency_ms, 2)},
            "steps": result["steps"],
            "citations": result["answer"].get("citations", []),
            "critiques": _critiques(result["answer"], serialized_evidence),
            "retrieved": trace_evidence,
            "agent_trace": result.get("_agent_trace", []),
            "execution_mode": "agent_graph" if use_agent_graph else "pipeline",
        }
    )
    return QueryResponse(
        answer=result["answer"],
        route=result["route"],
        metrics={**metrics, "latency_ms": round(latency_ms, 2)},
        steps=result["steps"],
        citations=result["answer"].get("citations", []),
        retrieved=serialized_evidence,
        critiques=_critiques(result["answer"], serialized_evidence),
        agent_trace=result.get("_agent_trace", []),
        execution_mode="agent_graph" if use_agent_graph else "pipeline",
    )


@app.post("/api/metrics")
def create_metrics(payload: MetricsCreateRequest):
    _ensure_state()
    row_id = app.state.metrics_store.insert_run(
        {**payload.model_dump(), **_strategy_snapshot(app.state.config)}
    )
    return {"id": row_id}


@app.get("/api/metrics", response_model=MetricsListResponse)
def list_metrics(limit: int = 200):
    _ensure_state()
    return MetricsListResponse(rows=app.state.metrics_store.list_runs(limit=limit))


@app.get("/api/traces")
def list_traces(limit: int = 100):
    _ensure_state()
    return {"rows": app.state.metrics_store.list_query_traces(limit)}


@app.post("/api/rag/setup")
def rag_setup(payload: RagSetupRequest):
    _ensure_state()
    if payload.vector_db != "chroma":
        raise HTTPException(status_code=400, detail="Only 'chroma' vector db is supported in this project.")
    if payload.chunk_overlap >= payload.chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be smaller than chunk_size.")

    if payload.use_saved_sample:
        documents = load_saved_sample()
    else:
        rows = load_dataset_subset(limit=payload.sample_limit, dataset_name=payload.dataset_name)
        documents = save_sample(rows, "data/sample")

    config = AppConfig.from_env()
    config.search_strategy = "chroma"
    config.rag_strategy = "auto"
    config.prompt_strategy = config.prompt_strategy
    config.answer_strategy = config.answer_strategy
    config.memory_strategy = config.memory_strategy
    config.evaluator_strategy = config.evaluator_strategy
    config.embedder_strategy = payload.embedder_strategy
    config.chunking_strategy = payload.chunking_strategy
    config.chunk_size = payload.chunk_size
    config.chunk_overlap = payload.chunk_overlap
    config.chroma_path = payload.vector_path
    config.chroma_collection = payload.vector_collection

    model_callable = None
    if config.answer_strategy == "llm":
        from src.generate.generate_answer import build_model_callable

        model_callable = build_model_callable()

    strategies = build_strategies_from_config(config, model_callable=model_callable)
    # Force index creation by executing one search call.
    if documents and documents[0].qa_pairs:
        run_pipeline(
            question=documents[0].qa_pairs[0].question,
            documents=documents,
            session_id="indexing",
            strategies=strategies,
        )

    app.state.config = config
    app.state.strategies = strategies
    app.state.model_callable = model_callable
    app.state.documents = documents
    _maybe_index_documents(config, strategies, documents)
    setup_id = app.state.metrics_store.insert_rag_setup(
        {
            "setup_name": payload.setup_name,
            "dataset_name": payload.dataset_name,
            "sample_limit": payload.sample_limit,
            "vector_db": payload.vector_db,
            "vector_path": payload.vector_path,
            "vector_collection": payload.vector_collection,
            "embedder_strategy": payload.embedder_strategy,
            "chunking_strategy": payload.chunking_strategy,
            "chunk_size": payload.chunk_size,
            "chunk_overlap": payload.chunk_overlap,
            "indexed_items": sum(len(item.qa_pairs) for item in documents),
        }
    )
    return {
        "setup_id": setup_id,
        "documents": len(documents),
        "qa_pairs": sum(len(item.qa_pairs) for item in documents),
    }


@app.get("/api/rag/setups")
def list_rag_setups(limit: int = 30):
    _ensure_state()
    return {"rows": app.state.metrics_store.list_rag_setups(limit)}


@app.post("/api/evaluate", response_model=EvaluationResponse)
def evaluate(payload: EvaluationRequest):
    _ensure_state()
    use_agent_graph = bool(app.state.config.use_multi_agent_graph)
    datapoints = []
    for document in app.state.documents:
        for pair in document.qa_pairs:
            datapoints.append((document.doc_id, pair.question, pair.answer, document.image_path or ""))
            if len(datapoints) >= payload.sample_size:
                break
        if len(datapoints) >= payload.sample_size:
            break
    if not datapoints:
        raise HTTPException(status_code=400, detail="No datapoints available for evaluation.")

    observations = []
    failure_counts: Counter[str] = Counter()
    correct = 0
    faithful = 0
    refused = 0
    cited = 0
    for doc_id, question, expected_answer, image_path in datapoints:
        if use_agent_graph:
            result = _run_agent_query(
                question=question,
                session_id=payload.session_id,
                expected_answer=expected_answer,
            )
        else:
            result = run_pipeline(
                question=question,
                documents=app.state.documents,
                session_id=payload.session_id,
                expected_answer=expected_answer,
                strategies=app.state.strategies,
            )
        predicted = result["answer"]["answer"]
        is_correct = predicted.strip().casefold() == expected_answer.strip().casefold()
        has_citation = bool(result["answer"].get("citations"))
        citation_match = has_citation and any(
            item.get("doc_id") == doc_id for item in result["answer"]["citations"]
        )
        is_faithful = bool(citation_match and not result["answer"].get("should_refuse"))
        category = _failure_category(
            expected_answer=expected_answer,
            predicted_answer=predicted,
            should_refuse=bool(result["answer"].get("should_refuse")),
            citations=result["answer"].get("citations", []),
            expected_doc_id=doc_id,
            retrieved_count=int(result["metrics"].get("retrieved_count", 0)),
        )
        note = (
            "Correct and grounded in cited doc."
            if is_correct and is_faithful
            else "Wrong answer text."
            if not is_correct
            else "Answer text matches but citation/refusal is weak."
        )
        failure_counts[category] += 1
        observations.append(
            {
                "question": question,
                "expected_answer": expected_answer,
                "predicted_answer": predicted,
                "doc_id": doc_id,
                "image_path": image_path,
                "image_url": _image_url(image_path),
                "is_correct": is_correct,
                "is_faithful": is_faithful,
                "failure_category": category,
                "note": note,
            }
        )
        correct += int(is_correct)
        faithful += int(is_faithful)
        refused += int(bool(result["answer"].get("should_refuse")))
        cited += int(has_citation)

    total = len(observations)
    summary = {
        "run_name": payload.run_name,
        "sample_size": total,
        "faithfulness": round(100 * faithful / total, 2),
        "answer_accuracy": round(100 * correct / total, 2),
        "refusal_rate": round(100 * refused / total, 2),
        "citation_rate": round(100 * cited / total, 2),
        "taxonomy_counts": dict(failure_counts),
    }
    eval_run_id = app.state.metrics_store.insert_eval_run(summary, observations)
    return EvaluationResponse(eval_run_id=eval_run_id, summary=summary, observations=observations)


@app.get("/api/evaluations")
def list_evaluations(limit: int = 20):
    _ensure_state()
    return {"rows": app.state.metrics_store.list_eval_runs(limit)}


@app.get("/api/evaluations/{eval_run_id}")
def get_evaluation(eval_run_id: int, limit: int = 300):
    _ensure_state()
    observations = app.state.metrics_store.list_eval_observations(eval_run_id, limit)
    for item in observations:
        item["image_url"] = _image_url(item.get("image_path", ""))
    return {
        "observations": observations,
    }


@app.post("/api/reports/retrieval")
def generate_retrieval_report(payload: RetrievalReportRequest):
    from scripts.evaluate_retrieval_strategies import evaluate_retrieval_strategies

    report = evaluate_retrieval_strategies(
        docs_path=payload.input_path,
        labels_path=payload.labels_path,
        output_path=payload.output_path,
    )
    if report is None:
        raise HTTPException(
            status_code=400,
            detail="No eligible QA pairs found in labels. Check label key fields.",
        )
    return {
        "output_path": payload.output_path,
        "summary_averages": report.get("summary_averages", {}),
        "strategies": list(report.get("detailed_results", {}).keys()),
        "question_count": {
            key: len(val)
            for key, val in report.get("detailed_results", {}).items()
        },
    }


@app.post("/api/reports/retrieval-trace")
def generate_retrieval_trace(payload: RetrievalTraceRequest):
    from scripts.debug_retrieval_reranker_trace import trace_retrieval_and_reranker

    report = trace_retrieval_and_reranker(
        docs_path=payload.input_path,
        labels_path=payload.labels_path,
        output_path=payload.output_path,
        text_output_path=payload.text_output_path,
        limit=payload.limit,
        table_rows=payload.table_rows,
        reranker_top_display=payload.reranker_top_display,
        index_docs=payload.index_docs,
    )
    return {
        "output_path": payload.output_path,
        "text_output_path": payload.text_output_path,
        "items": len(report.get("items", [])),
        "meta": report.get("meta", {}),
    }


@app.post("/api/reports/trace-analysis")
def generate_trace_analysis(payload: TraceAnalysisRequest):
    from scripts.retrieve_analysis_script import analyze_trace, format_report

    report = analyze_trace(payload.input_path)
    summary_text = format_report(report)

    output_json = Path(payload.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if payload.text_output:
        text_path = Path(payload.text_output)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(summary_text, encoding="utf-8")

    return {
        "output_json": payload.output_json,
        "text_output": payload.text_output,
        "summary": {
            "total_questions": report.get("total_questions", 0),
            "complete_failure_count": report.get("complete_failure_count", 0),
            "partial_failure_count": report.get("partial_failure_count", 0),
            "reranker_improvement_count": report.get("reranker_improvement_count", 0),
            "reranker_regression_count": report.get("reranker_regression_count", 0),
        },
    }