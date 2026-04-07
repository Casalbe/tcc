import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or to a .env file."
        )

    _client = OpenAI(api_key=api_key)
    return _client


def generate_chatgpt_review(prompt: str, model: str = "gpt-4.1", temperature: float = 0.2) -> str:
    response = _get_client().chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content
    return (content or "").strip()
