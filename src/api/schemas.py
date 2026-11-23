"""Request and response schemas for the FastAPI service."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str = "api-default"
    run_name: str = "interactive"
    scope: str = "global"
    doc_id: str | None = None
    use_agent_graph: bool | None = None


class QueryResponse(BaseModel):
    answer: dict
    route: str
    metrics: dict
    steps: list[str]
    citations: list[dict]
    retrieved: list[dict]
    critiques: list[str]
    agent_trace: list[str] = []
    execution_mode: str = "pipeline"


class MetricsCreateRequest(BaseModel):
    run_name: str = "manual"
    question: str
    answer: str
    route: str = "text"
    retrieved_count: int = 0
    has_citation: bool = False
    refused: bool = False
    exact_match: float | None = None
    latency_ms: float = 0.0


class MetricsListResponse(BaseModel):
    rows: list[dict]


class RagSetupRequest(BaseModel):
    setup_name: str = "manual-setup"
    dataset_name: str = "sujet-ai/Sujet-Finance-QA-Vision-100k"
    sample_limit: int = Field(default=10, ge=1, le=200)
    vector_db: str = "chroma"
    vector_path: str = "data/processed/chroma"
    vector_collection: str = "finance_sample"
    embedder_strategy: str = "hashing"
    chunking_strategy: str = "whole"
    chunk_size: int = Field(default=800, ge=50, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=2000)
    use_saved_sample: bool = False


class EvaluationRequest(BaseModel):
    run_name: str = "eval-run"
    sample_size: int = Field(default=100, ge=1, le=500)
    session_id: str = "eval-session"


class EvaluationResponse(BaseModel):
    eval_run_id: int
    summary: dict
    observations: list[dict]


class RetrievalReportRequest(BaseModel):
    input_path: str = "data/sample_500/documents.json"
    labels_path: str = "data/evaluation/500_docs_parent_child/evaluation_gold_set_200.json"
    output_path: str = "data/evaluation/500_docs_parent_child/retrieval_evaluation_results_with_reranker.json"


class RetrievalTraceRequest(BaseModel):
    input_path: str = "data/sample_500/documents.json"
    labels_path: str = "data/evaluation/500_docs_parent_child/evaluation_gold_set_200.json"
    output_path: str = "data/evaluation/500_docs_parent_child/reports/retrieval_reranker_trace_200.json"
    text_output_path: str | None = "data/evaluation/500_docs_parent_child/reports/retrieval_reranker_trace_200.txt"
    limit: int = Field(default=25, ge=1, le=500)
    table_rows: int = Field(default=30, ge=5, le=200)
    reranker_top_display: int = Field(default=10, ge=1, le=100)
    index_docs: bool = False


class TraceAnalysisRequest(BaseModel):
    input_path: str = "data/evaluation/500_docs_parent_child/reports/retrieval_reranker_trace_200.json"
    output_json: str = "data/evaluation/500_docs_parent_child/reports/trace_analysis_report_200.json"
    text_output: str | None = "data/evaluation/500_docs_parent_child/reports/retrieve_analysis_summary_200.txt"
