import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your environment or to a .env file."
        )

    _client = anthropic.Anthropic(api_key=api_key)
    return _client


def generate_claude_review(
    prompt: str,
    model: str = "claude-3-7-sonnet-20250219",
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> str:
    response = _get_client().messages.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    blocks = getattr(response, "content", None) or []
    if not blocks:
        return ""

    first = blocks[0]
    text = getattr(first, "text", None)
    if isinstance(text, str):
        return text.strip()
    return str(first).strip()
