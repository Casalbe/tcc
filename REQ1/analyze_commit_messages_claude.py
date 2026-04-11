import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0.2

#tenta carregar o .env para obter o token do claude, caso nao exista o token deve ser passado por variavel de ambiente ou CLI
def load_dotenv(dotenv_path: Path) -> None:

    if not dotenv_path.exists() or not dotenv_path.is_file():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


#prompt
def build_prompt(commit_message: str, diff: str) -> str:
    return (
        "Você é um analista de código profissional, analise a mensagem de commit a seguir:\n\n"
        f"{commit_message}\n\n"
        "E o seu diff correspondente:\n\n"
        f"{diff}\n\n"
        "Baseado neles eu quero que você responda o seguinte:\n\n"
        "A mensagem de commit reflete realmente a mudança de código feita? "
        "Classifique em Sim, Provavelmente sim, Provavelmente não, Não, Sem mensagem.\n\n"
        "Dê uma nota, precisa, de 0 a 100 para a qualidade da mensagem do commit\n\n"
        "A sua mensagem deve conter apenas \"(reflete_mudanca, qualidade)\" "
        "onde reflete mudança é a classificação da reflexão da mudança na mensagem "
        "e qualidade a sua nota dada a qualidade da mensagem"
    )


_RE_PAREN = re.compile(r"\((.*?)\)")


#função para extrair a resposta do claude no formato (reflete_mudanca, qualidade) onde reflete_mudanca é a classificação da reflexão da mudança na mensagem e qualidade a nota dada a qualidade da mensagem
def parse_tuple_response(text: str) -> Tuple[str, int]:

    m = _RE_PAREN.search(text.strip())
    if not m:
        raise ValueError(f"Could not find '(...)' tuple in response: {text!r}")

    inner = m.group(1)
    # split on first comma
    if "," not in inner:
        raise ValueError(f"Could not split tuple by comma: {text!r}")

    left, right = inner.split(",", 1)
    reflete = left.strip()

    # Extract first integer in right
    m_num = re.search(r"-?\d+", right)
    if not m_num:
        raise ValueError(f"Could not parse quality integer from: {text!r}")

    quality = int(m_num.group(0))
    if quality < 0:
        quality = 0
    if quality > 100:
        quality = 100

    return reflete, quality


#chamada para a API do claude, retornando a resposta como string
def call_claude(api_key: str, prompt: str, timeout_s: int = 60) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": MODEL,
        "max_tokens": 200,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }

    resp = requests.post(CLAUDE_API_URL, headers=headers, json=payload, timeout=timeout_s)
    if resp.status_code in {401, 403}:
        raise RuntimeError(
            f"Claude API auth error ({resp.status_code}). Check your ANTHROPIC_API_KEY/CLAUDE_API_KEY."
        )
    resp.raise_for_status()
    data = resp.json()

    # Anthropic returns: { content: [ {type:'text', text:'...'} ], ... }
    parts = data.get("content") or []
    texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text"]
    return "".join(texts).strip()


def load_input_items(path: Path) -> List[Dict[str, Any]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("Input JSON must be a list of objects")
    return items


def load_existing_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Output JSON exists but could not be parsed: {path} ({e})")

    if not data:
        return []
    if not isinstance(data, list):
        raise ValueError("Output JSON must be a list of objects")
    return data


def write_results(path: Path, results: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze commit messages vs diffs using Claude.")
    p.add_argument("--input", "-i", required=True, help="Input JSON from scraper")
    p.add_argument("--output", "-o", required=True, help="Output JSON with Claude results")
    p.add_argument(
        "--limit",
        type=int,
        default=2,
        help="How many commits to send to Claude (default: 2)",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Claude API key (default: from .env/env vars)",
    )
    p.add_argument(
        "--begin-at",
        type=int,
        default=1,
        help=(
            "1-based index in the input JSON list to start from (default: 1). "
            "Useful to resume after rate limits."
        ),
    )

    args = p.parse_args()

    # Load .env from repo root
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    load_dotenv(repo_root / ".env")

    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing Claude API key. Set ANTHROPIC_API_KEY (or CLAUDE_API_KEY) in .env.")

    in_path = Path(args.input)
    out_path = Path(args.output)

    items = load_input_items(in_path)

    if args.begin_at < 1:
        raise ValueError("--begin-at must be >= 1")

    results: List[Dict[str, Any]] = load_existing_results(out_path)
    existing_hashes = {
        (r.get("commit_hash") or "").strip()
        for r in results
        if isinstance(r, dict) and (r.get("commit_hash") or "").strip()
    }

    if results:
        log(f"Loaded {len(results)} existing results from {out_path}")
    log(f"Input has {len(items)} items; starting at index {args.begin_at}")

    sent = 0

    for idx, obj in enumerate(items, start=1):
        if idx < args.begin_at:
            continue
        if sent >= args.limit:
            break

        commit_hash = (obj.get("commit_hash") or "").strip()
        commit_message = obj.get("commit_message") or ""
        diff = obj.get("diff") or ""

        if not commit_hash or not diff:
            log(f"Skip item {idx}: missing commit_hash or diff")
            continue

        if commit_hash in existing_hashes:
            log(f"Skip item {idx}: already in output ({commit_hash})")
            continue

        sent += 1
        log(f"Calling Claude {sent}/{args.limit} for commit {commit_hash}")

        prompt = build_prompt(commit_message, diff)
        text = call_claude(api_key=api_key, prompt=prompt)

        try:
            reflete_mudanca, qualidade = parse_tuple_response(text)
        except Exception as e:
            log(f"Parse error for {commit_hash}: {e}; raw={text!r}")
            reflete_mudanca, qualidade = "PARSE_ERROR", -1

        results.append(
            {
                "commit_hash": commit_hash,
                "reflete_mudanca": reflete_mudanca,
                "qualidade": qualidade,
                "raw": text,
            }
        )
        existing_hashes.add(commit_hash)

        # Persist after each request so progress isn't lost if rate-limited.
        write_results(out_path, results)
        log(f"Saved progress: {len(results)} results -> {out_path}")

    log(f"Done. Wrote {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
