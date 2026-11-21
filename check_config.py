"""Quick config & environment sanity check from project root."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

env_path = find_dotenv(usecwd=True)
loaded = load_dotenv(env_path, override=False)

from src.config import AppConfig  # noqa: E402

cfg = AppConfig()

raw_key = os.getenv("EMBEDDING_API_KEY") or ""
cfg_key = getattr(cfg, "embedding_api_key", None) or getattr(cfg, "api_key", None) or ""
ca_cert = os.getenv("EMBEDDING_CA_CERT") or getattr(cfg, "embedding_ca_cert", None) or ""


def mask_secret(val: str) -> str:
    val = val.strip()
    if not val:
        return "<MISSING>"
    if len(val) <= 6:
        return f"<SET, len={len(val)}>"
    return f"{val[:3]}...{val[-3:]} (len={len(val)})"


print("=== Environment & .env Check ===")
print(f"CWD                   : {Path.cwd()}")
print(f".env found            : {env_path or '<NOT FOUND>'}")
print(f"load_dotenv() loaded  : {loaded}")
print(f"os.environ KEY        : {mask_secret(raw_key)}")
print(f"AppConfig KEY (if any): {mask_secret(str(cfg_key))}")
print(f"EMBEDDING_API_BASE_URL: {os.getenv('EMBEDDING_API_BASE_URL') or getattr(cfg, 'embedding_api_base_url', '<UNSET>')}")
print(f"EMBEDDING_CA_CERT     : {ca_cert or '<UNSET>'}" + (f" (exists={Path(ca_cert).exists()})" if ca_cert else ""))

print("\n=== AppConfig Values ===")
for field in (
    "chunking_strategy",
    "chunk_size",
    "chunk_overlap",
    "embedder_strategy",
    "embedding_model_name",
    "search_strategy",
    "retrieval_limit",
    "chroma_path",
    "chroma_collection",
    "reranker_model",
):
    if hasattr(cfg, field):
        print(f"{field:<22}: {getattr(cfg, field)}")