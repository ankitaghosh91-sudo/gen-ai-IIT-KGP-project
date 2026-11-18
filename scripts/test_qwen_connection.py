
import os

from openai import OpenAI
from src.config import AppConfig


AppConfig.from_env()

api_key = os.getenv("QWEN_API_KEY")
base_url = os.getenv("QWEN_BASE_URL")
model = os.getenv("QWEN_MODEL")

if not all([api_key, base_url, model]):
    raise ValueError("Qwen API configuration is incomplete.")

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    timeout=30.0,
    max_retries=0,
)
print("\nEnvironment base URL:", repr(base_url))
print("\nClient base URL:", repr(str(client.base_url)))
print("\nAPI key loaded:", bool(api_key))

print("\nAPI key length:", len(api_key))
print("\nAPI key has leading/trailing whitespace:", api_key != api_key.strip())
print("\nAPI key starts with quote:", api_key.startswith(("'", '"')))
print("API key ends with quote:", api_key.endswith(("'", '"')))

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: Qwen endpoint is responding.",
            }
        ],
        max_tokens=30,
        temperature=0,
    )

    print("Qwen API request successful.")
    print("Model:", response.model)
    print("Response:", response.choices[0].message.content)

except Exception as exc:
    print("Error type:", type(exc).__name__)
    print("Error:", str(exc))

    cause = exc.__cause__

    if cause:
        print("Underlying error type:", type(cause).__name__)
        print("Underlying error:", repr(cause))
    else:
        print("No underlying exception available.")
