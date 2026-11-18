
"""Provider-independent LLM client for evaluation tasks."""

import os
import httpx

from openai import OpenAI
from src.config import AppConfig


class LLMClient:

    def __init__(self):
        AppConfig.from_env()

        self.provider = os.getenv("LLM_PROVIDER", "groq").lower()

        if self.provider == "groq":
            from groq import Groq

            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY is missing.")

            self.model = os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-20b",
            )

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

            self.model = os.getenv(
                "QWEN_MODEL",
                "Qwen/Qwen3-8B",
            )

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
            raise ValueError(
                f"Unsupported LLM provider: {self.provider}"
            )

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 500,
        response_format: dict | None = None,
    ) -> str:
        """Generate a text response from the configured LLM."""

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
            raise ValueError(
                f"LLM returned an empty response "
                f"(finish_reason={choice.finish_reason!r})"
            )

        return content.strip()
