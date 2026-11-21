from typing import Any, Literal
from langgraph.graph import END, START, StateGraph

from src.agents.state import AgentRAGState, TaskPlan


def supervisor_node(state: AgentRAGState) -> dict[str, Any]:
    """Plans tasks, manages retry budget, and decides whether to continue or stop."""
    current_attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 2)
    trace = list(state.get("trace", []))

    if state.get("verified", False):
        trace.append("supervisor: verified -> finalize")
        plan: TaskPlan = {
            "mode": "finalize",
            "focus_gaps": [],
            "feedback": "",
        }
        return {"plan": plan, "trace": trace}

    if current_attempt >= max_attempts:
        trace.append(f"supervisor: max_attempts ({max_attempts}) reached -> finalize")
        plan = {
            "mode": "finalize",
            "focus_gaps": state.get("evidence_gaps", []),
            "feedback": state.get("verification_notes", "Retry limit reached."),
        }
        return {"plan": plan, "trace": trace}

    next_attempt = current_attempt + 1
    mode: Literal["initial_search", "retry_search"] = (
        "initial_search" if next_attempt == 1 else "retry_search"
    )
    gaps = state.get("evidence_gaps", [])
    feedback = state.get("verification_notes", "")

    trace.append(f"supervisor: attempt={next_attempt}/{max_attempts} mode={mode}")
    plan = {
        "mode": mode,
        "focus_gaps": gaps,
        "feedback": feedback,
    }
    return {
        "attempt": next_attempt,
        "plan": plan,
        "trace": trace,
    }


def finalize_node(state: AgentRAGState) -> dict[str, Any]:
    """Emits the final response payload with citations and execution trace."""
    trace = list(state.get("trace", []))
    trace.append("finalize: emitted final_output")
    answer = state.get("draft_answer") or (
        "Unable to find sufficient verified evidence in the retrieved documents."
    )
    citations = state.get("citations", []) if state.get("draft_answer") else []

    return {
        "trace": trace,
        "final_output": {
            "answer": answer,
            "citations": citations,
            "verified": state.get("verified", False),
            "attempts_used": state.get("attempt", 0),
            "trace": trace,
        },
    }


def route_from_supervisor(state: AgentRAGState) -> Literal["query_planning", "finalize"]:
    mode = state.get("plan", {}).get("mode", "finalize")
    return "finalize" if mode == "finalize" else "query_planning"


def build_step1_graph():
    """Standalone graph for testing Step 1 before wiring Step 2."""
    graph = StateGraph(AgentRAGState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "query_planning": "finalize",
            "finalize": "finalize",
        },
    )
    graph.add_edge("finalize", END)
    return graph.compile()