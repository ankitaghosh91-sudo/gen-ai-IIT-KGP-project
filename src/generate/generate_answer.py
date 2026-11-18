
"""Factory for creating the LLM answer model callable."""

from src.evaluation.llm_client import LLMClient
from src.generate.llm_adapter import LLMModelAdapter


def build_model_callable():
    """Create the configured LLM model callable."""
    llm_client = LLMClient()
    return LLMModelAdapter(llm_client)
