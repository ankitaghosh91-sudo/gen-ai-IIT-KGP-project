
import os

from groq import Groq
from src.config import AppConfig


AppConfig.from_env()

api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL")

if not api_key or not model:
    raise ValueError("Groq API configuration is incomplete.")

client = Groq(
    api_key=api_key,
    timeout=30.0,
    max_retries=0,
)



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
                "content": "Reply with exactly: Groq endpoint is responding.",
            }
        ],
        max_tokens=50,
        temperature=0,
    )

    print("Groq API request successful.")
    print("Model:", response.model)
    print("Response:", response.choices[0].message.content)

except Exception as exc:
    print("Groq API request failed.")
    print("Error type:", type(exc).__name__)
    print("Error:", str(exc))
