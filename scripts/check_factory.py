from src.config import AppConfig
from src.pipeline import build_strategies_from_config
from src.strategies.search import ChromaSearch

config = AppConfig()

print("Configuration loaded:")
print("Search strategy:", config.search_strategy)
print("Chunking strategy:", config.chunking_strategy)
print("Embedder strategy:", config.embedder_strategy)
print("Embedding model:", config.embedding_model_name)
print("Retrieval limit:", config.retrieval_limit)

strategies = build_strategies_from_config(config)

print("\nFactory output:")

rag = strategies.rag

print("RAG implementation:", type(rag).__name__)

if hasattr(rag, "text"):
    rag = rag.text

search = rag.search

# Unwrap LimitedSearch, if present.
while hasattr(search, "base"):
    search = search.base

print("Search implementation:", type(search).__name__)

if isinstance(search, ChromaSearch):
    print("Embedder implementation:", type(search.embedder).__name__)
    print("Chunking implementation:", type(search.chunker).__name__)
