"""Provider-independent LLM client for evaluation and RAG tasks."""

import logging
import os
import httpx

from openai import OpenAI
from src.config import AppConfig

logger = logging.getLogger(__name__)


def _should_fallback(exc: Exception) -> bool:
    exc_type = type(exc).__name__.lower()
    msg = str(exc).lower()
    status_code = getattr(exc, "status_code", None)

    if status_code in (400, 408, 429, 500, 502, 503, 504):
        return True

    if "badrequesterror" in exc_type or "apistatuserror" in exc_type or "apiconnectionerror" in exc_type:
        return True

    return any(
        token in msg
        for token in (
            "error code: 400",
            "status code: 400",
            "maximum context length",
            "input_tokens",
            "context_length_exceeded",
            "context length",
            "too many tokens",
            "429",
            "rate limit",
            "timeout",
            "timed out",
            "500",
            "502",
            "503",
            "504",
        )
    )


def _trim_messages_for_fallback(messages: list[dict], max_chars: int = 12000) -> list[dict]:
    """Keep system message intact and trim user/assistant content if very long."""
    if not messages:
        return messages

    trimmed: list[dict] = []
    remaining = max_chars

    for msg in messages:
        if msg.get("role") == "system":
            trimmed.append(dict(msg))

    for msg in reversed(messages):
        if msg.get("role") == "system":
            continue
        content = str(msg.get("content", ""))
        if len(content) <= remaining:
            trimmed.append({**msg, "content": content})
            remaining -= len(content)
        elif remaining > 200:
            trimmed.append({**msg, "content": content[:remaining]})
            remaining = 0

    system_msgs = [m for m in trimmed if m.get("role") == "system"]
    non_system_msgs = [m for m in trimmed if m.get("role") != "system"]
    non_system_msgs.reverse()
    return system_msgs + non_system_msgs


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        *,
        enable_fallback: bool = True,
    ):
        AppConfig.from_env()

        self.provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()

        if self.provider == "groq":
            from groq import Groq

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY is missing.")

            self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

            self.client = Groq(
                api_key=api_key,
                timeout=30.0,
                max_retries=2,
            )

        elif self.provider == "qwen":
            api_key = os.getenv("QWEN_API_KEY")
            base_url = os.getenv("QWEN_BASE_URL")
            ca_cert = os.getenv("QWEN_CA_CERT")

            if not api_key or not base_url or not ca_cert:
                raise ValueError(
                    "Qwen API configuration is incomplete. "
                    "Check QWEN_API_KEY, QWEN_BASE_URL, and QWEN_CA_CERT."
                )

            if not os.path.isfile(ca_cert):
                raise FileNotFoundError(
                    f"Qwen CA certificate not found: {ca_cert}"
                )

            self.model = model or os.getenv("QWEN_MODEL", "Qwen/Qwen3-8B")

            http_client = httpx.Client(
                verify=ca_cert,
                timeout=30.0,
            )

            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=http_client,
                max_retries=2,
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

        self.fallback_client: "LLMClient | None" = None
        if enable_fallback:
            self.fallback_client = self._build_fallback_client()

        logger.info(
            "LLM_CLIENT_READY | provider=%s | model=%s | fallback_enabled=%s | fallback_model=%s",
            self.provider,
            self.model,
            self.fallback_client is not None,
            self.fallback_client.model if self.fallback_client else None,
        )

    def _build_fallback_client(self) -> "LLMClient | None":
        fallback_provider = os.getenv("FALLBACK_LLM_PROVIDER", "").strip().lower()
        fallback_model = os.getenv("FALLBACK_LLM_MODEL", "").strip() or None

        if not fallback_provider and not fallback_model:
            return None

        target_provider = fallback_provider or self.provider
        if target_provider == self.provider and fallback_model == self.model:
            return None

        try:
            return LLMClient(
                provider=target_provider,
                model=fallback_model,
                enable_fallback=False,
            )
        except Exception as exc:
            logger.warning("FALLBACK_LLM_INIT_FAILED | error=%s", exc)
            return None

    def _call_provider(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        response_format: dict | None,
        *,
        _retry_on_length: bool = True,
    ) -> str:
        request_args = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format is not None:
            request_args["response_format"] = response_format

        response = self.client.chat.completions.create(**request_args)

        choice = response.choices[0]
        content = choice.message.content

        if not content:
            if choice.finish_reason == "length" and _retry_on_length:
                bumped_tokens = min(max(max_tokens * 2, 1500), 3000)
                logger.warning(
                    "LLM_EMPTY_ON_LENGTH | provider=%s | model=%s | retrying max_tokens=%d",
                    self.provider,
                    self.model,
                    bumped_tokens,
                )
                return self._call_provider(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=bumped_tokens,
                    response_format=response_format,
                    _retry_on_length=False,
                )

            raise ValueError(
                f"LLM returned an empty response "
                f"(finish_reason={choice.finish_reason!r})"
            )

        return content.strip()

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 500,
        response_format: dict | None = None,
    ) -> str:
        """Generate a text response, falling back to a secondary LLM on context/transient errors."""
        try:
            return self._call_provider(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except Exception as exc:
            body = getattr(exc, "body", None)
            if self.fallback_client is None or not _should_fallback(exc):
                logger.error(
                    "LLM_PRIMARY_FAILED_NO_FALLBACK | provider=%s | model=%s | has_fallback=%s | body=%r",
                    self.provider,
                    self.model,
                    self.fallback_client is not None,
                    body,
                )
                raise

            logger.warning(
                "LLM_FALLBACK_TRIGGERED | primary_provider=%s | primary_model=%s | "
                "fallback_provider=%s | fallback_model=%s | body=%r | error=%s",
                self.provider,
                self.model,
                self.fallback_client.provider,
                self.fallback_client.model,
                body,
                exc,
            )

            trimmed_messages = _trim_messages_for_fallback(messages)
            fallback_max_tokens = min(max(max_tokens, 800), 1200)
            try:
                return self.fallback_client._call_provider(
                    messages=trimmed_messages,
                    temperature=temperature,
                    max_tokens=fallback_max_tokens,
                    response_format=response_format,
                )
            except Exception as fallback_exc:
                logger.error(
                    "LLM_FALLBACK_FAILED | fallback_provider=%s | fallback_model=%s | body=%r | error=%s",
                    self.fallback_client.provider,
                    self.fallback_client.model,
                    getattr(fallback_exc, "body", None),
                    fallback_exc,
                )
                raise