import os

from dotenv import load_dotenv

load_dotenv()


def _get_github_token() -> str:
    token = (
        os.getenv("GITHUB_TOKEN")
        or os.getenv("GITHUB_TOKEN_ENV")
        or os.getenv("GH_TOKEN")
        or os.getenv("GITHUB_PAT")
    )
    return (token or "").strip()


GITHUB_TOKEN = _get_github_token()

HEADERS: dict[str, str] = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

# Avoid sending an empty/invalid Authorization header.
if GITHUB_TOKEN:
    HEADERS["Authorization"] = f"Bearer {GITHUB_TOKEN}"