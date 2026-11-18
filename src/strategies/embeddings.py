
"""Embedding strategies."""

import hashlib
import math

import numpy as np
import os
import time
import requests

from src.strategies.base import EmbedderBase

import logging
import time

logger = logging.getLogger(__name__)


class HashingEmbedder(EmbedderBase):
    """Tiny offline encoder for a demo."""

    def __init__(self, dimensions: int = 128):
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("dimensions must be a positive integer")

        self.dimensions = dimensions

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")

        vectors = []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"Item at index {index} must be a string")

            vector = [0.0] * self.dimensions

            for word in text.lower().split():
                slot = int(hashlib.sha1(word.encode()).hexdigest(), 16) % self.dimensions
                vector[slot] += 1.0

            length = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / length for value in vector])

        return vectors


class SentenceTransformerEmbedder(EmbedderBase):
    """Sentence Transformer encoder with batching and query support."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ):
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")

        if not isinstance(normalize_embeddings, bool):
            raise TypeError("normalize_embeddings must be a boolean")

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        # Lazy loading: model is loaded only when embeddings are requested.
        self.model = None
        self.dimension = None

    def load_model(self) -> None:
        """Load the model once and determine its embedding dimension."""

        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            self.dimension =  self.model.get_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a batch of document chunks."""

        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"Item at index {index} must be a string "
                    f"(got {type(text).__name__})"
                )

            if not text.strip():
                raise ValueError(
                    f"Item at index {index} is empty or whitespace-only"
                )

        self.load_model()

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a single user query."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        self.load_model()

        embedding = self.model.encode(
            query,
            batch_size=1,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return np.asarray(embedding, dtype=np.float32)

    def get_dimensions(self) -> int:
        """Return the embedding vector dimension."""

        self.load_model()
        return self.dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Implement the EmbedderBase interface."""

        return self.embed_texts(texts).tolist()



class BGEAPIEmbedder(EmbedderBase):
    """BGE-small-en-v1.5 embedding API client."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str | None = None,
        batch_size: int = 16,
        ca_cert: str | None = None,
    ):
        self.base_url = (
            base_url or os.getenv(
                "EMBEDDING_API_BASE_URL",
                "https://103.42.50.41:1035/v1",
            )
        ).rstrip("/")

        self.model_name = (
            model_name or os.getenv(
                "EMBEDDING_MODEL",
                "bge-small-en-v1.5",
            )
        )

        self.api_key = os.getenv("EMBEDDING_API_KEY")

        self.ca_cert = (
            ca_cert or os.getenv("EMBEDDING_CA_CERT")
        )

        if not self.api_key:
            raise ValueError("EMBEDDING_API_KEY is missing")

        if not self.ca_cert:
            raise ValueError("EMBEDDING_CA_CERT is missing")

        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= 16
        ):
            raise ValueError("batch_size must be between 1 and 16")

        self.batch_size = batch_size
        self.dimension = 384

        self.session = requests.Session()

    def _embed(
        self,
        texts: list[str],
        is_query: bool = False,
    ) -> list[list[float]]:

        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")

        if not texts:
            return []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"Item at index {index} must be a string"
                )

            if not text.strip():
                raise ValueError(
                    f"Item at index {index} is empty"
                )

        if is_query:
            texts = [
                "Represent this sentence for searching relevant passages: "
                + text
                for text in texts
            ]

        embeddings = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]

            payload = {
                "model": self.model_name,
                "input": batch,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            logger.info(
                "BGE_API_REQUEST | model=%s | batch_size=%d | is_query=%s",
                self.model_name,
                len(batch),
                is_query,
            )
            for attempt in range(3):
                try:
                    response = self.session.post(
                        f"{self.base_url}/embeddings",
                        json=payload,
                        headers=headers,
                        verify=self.ca_cert,
                        timeout=60,
                    )

                    if response.status_code in (429, 502, 503, 504):
                        if attempt < 2:
                            time.sleep(2 * (attempt + 1))
                            continue

                    response.raise_for_status()
                    
                    break

                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))

            data = response.json()["data"]

            data.sort(key=lambda item: item["index"])

            batch_embeddings = [
                item["embedding"] for item in data
            ]

            logger.info(
                "BGE_API_RESPONSE | model=%s | returned=%d | dimensions=%d",
                self.model_name,
                len(batch_embeddings),
                len(batch_embeddings[0]) if batch_embeddings else 0,
            )
    
            if len(batch_embeddings) != len(batch):
                raise ValueError(
                    "Embedding API returned an unexpected number of vectors"
                )

            for embedding in batch_embeddings:
                if len(embedding) != self.dimension:
                    raise ValueError(
                        f"Expected {self.dimension} dimensions, "
                        f"received {len(embedding)}"
                    )

            embeddings.extend(batch_embeddings)

        return embeddings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks without query instructions."""

        return self._embed(texts, is_query=False)

    def embed_query(self, query: str) -> list[float]:
        """Embed a user query using the BGE retrieval instruction."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        return self._embed([query], is_query=True)[0]
    

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Embed multiple user queries using the BGE retrieval instruction."""

        return self._embed(queries, is_query=True)

    def get_dimensions(self) -> int:
        return self.dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Implement EmbedderBase interface."""

        return self.embed_texts(texts)

