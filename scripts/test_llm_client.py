
from src.evaluation.llm_client import LLMClient


def test_llm_client():

    llm = LLMClient()

    print("Provider:", llm.provider)
    print("Model:", llm.model)

    response = llm.generate(
        messages=[
            {
                "role": "user",
                "content": "Reply with exactly: LLM client abstraction works.",
            }
        ]
    )

    print("Response:", response)


if __name__ == "__main__":
    test_llm_client()
