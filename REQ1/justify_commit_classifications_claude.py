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


# tenta carregar o .env para obter o token do claude
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


def _to_claude_class(reflete_mudanca: str) -> str:
    """Normalize label to the four classes in the prompt (keeping Portuguese).

    Input may contain accents or variations (e.g. 'Não', 'Provavelmente não').
    """
    if reflete_mudanca is None:
        return ""

    s = str(reflete_mudanca).strip().lower()
    s = " ".join(s.split())

    mapping = {
        "sim": "Sim",
        "provavelmente sim": "Provavelmente sim",
        "nao": "Não",
        "não": "Não",
        "provavelmente nao": "Provavelmente não",
        "provavelmente não": "Provavelmente não",
    }

    return mapping.get(s, str(reflete_mudanca).strip())


def build_prompt(diff: str, commit_message: str, reflete_mudanca: str, qualidade: Any) -> str:
    # Keep the user's prompt structure; insert data verbatim.
    # Note: We use the correct spelling 'Provavelmente não' in the list.
    claude_class = _to_claude_class(reflete_mudanca)

    return (
        "Você, como um analista de codigo profissional, analisou o seguinte diff de um commit:\n\n"
        f"{diff}\n\n"
        "E a mensagem do commit:\n\n"
        f"{commit_message}\n\n"
        f"E classificou o quanto a mensagem reflete a mudança do codigo como \"{claude_class}\", das seguintes classificações:\n\n"
        "Sim\n"
        "Provavelmente sim\n"
        "Provavelmente não\n"
        "Não\n\n"
        f"E deu a nota {qualidade}, dentro do alcance 0-100, referente a qualidade da mensagem do commit.\n\n"
        "Eu quero que você justifique as duas classificações que fez em uma resposta concisa e precisa, no seguinte formato:\n\n"
        "Justificativa para a classificação: [sua justificativa aqui]\n"
        "Justificativa para a nota: [sua justificativa aqui]"
    )


_RE_CLASS = re.compile(
    r"justificativa\s*para\s*a\s*classifica[cç][aã]o\s*:\s*",
    flags=re.IGNORECASE,
)
_RE_NOTE = re.compile(
    r"justificativa\s*para\s*a\s*nota\s*:\s*",
    flags=re.IGNORECASE,
)


def parse_justification_response(text: str) -> Tuple[str, str]:
    """Extract two separate justifications from Claude output.

    Expected format:
      Justificativa para a classificação: ...
      Justificativa para a nota: ...

    Returns (class_justification, quality_justification). If parsing fails,
    returns ("", "") so the caller can still persist raw output.
    """
    if not text:
        return "", ""

    m_class = _RE_CLASS.search(text)
    m_note = _RE_NOTE.search(text)
    if not m_class and not m_note:
        return "", ""

    if m_class and m_note:
        if m_class.end() <= m_note.start():
            class_part = text[m_class.end() : m_note.start()].strip()
            note_part = text[m_note.end() :].strip()
        else:
            # Unlikely ordering, but handle gracefully.
            note_part = text[m_note.end() : m_class.start()].strip()
            class_part = text[m_class.end() :].strip()
        return class_part, note_part

    # If only one marker is present, try a line-based fallback.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    class_part = ""
    note_part = ""
    for ln in lines:
        low = ln.lower()
        if "justificativa" in low and "classifica" in low and ":" in ln:
            class_part = ln.split(":", 1)[1].strip()
        elif "justificativa" in low and "nota" in low and ":" in ln:
            note_part = ln.split(":", 1)[1].strip()

    return class_part, note_part


def call_claude(api_key: str, prompt: str, timeout_s: int = 60) -> str:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": MODEL,
        "max_tokens": 250,
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

    parts = data.get("content") or []
    texts = [
        p.get("text", "")
        for p in parts
        if isinstance(p, dict) and p.get("type") == "text"
    ]
    return "".join(texts).strip()


def load_input_rows(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    # Accept either:
    # - {"rows": [...], "summary": {...}}
    # - a raw list of rows
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        rows = data["rows"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError("Input JSON must be a list of rows or an object with a 'rows' array")

    if not all(isinstance(r, dict) for r in rows):
        raise ValueError("All rows in input JSON must be objects")

    return rows


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


def _make_key(repo: Any, commit_hash: Any) -> str:
    return f"{str(repo).strip()}::{str(commit_hash).strip()}"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate concise justifications for existing reflete_mudanca + qualidade labels using Claude."
    )
    p.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input JSON produced by sample_commits_by_class.py",
    )
    p.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output JSON with justifications",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of commits to send to Claude in this run (0 = no limit)",
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
            "1-based index in the input rows list to start from (default: 1). "
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
        raise RuntimeError(
            "Missing Claude API key. Set ANTHROPIC_API_KEY (or CLAUDE_API_KEY) in .env."
        )

    in_path = Path(args.input)
    out_path = Path(args.output)

    rows = load_input_rows(in_path)

    if args.begin_at < 1:
        raise ValueError("--begin-at must be >= 1")

    results: List[Dict[str, Any]] = load_existing_results(out_path)
    existing_keys = {
        _make_key(r.get("repo"), r.get("commit_hash"))
        for r in results
        if isinstance(r, dict) and r.get("repo") and r.get("commit_hash")
    }

    if results:
        log(f"Loaded {len(results)} existing results from {out_path}")
    log(f"Input has {len(rows)} rows; starting at index {args.begin_at}")

    sent = 0

    for idx, row in enumerate(rows, start=1):
        if idx < args.begin_at:
            continue
        if args.limit and sent >= args.limit:
            break

        if not isinstance(row, dict):
            log(f"Skip row {idx}: not an object")
            continue

        repo = row.get("repo")
        commit_hash = row.get("commit_hash")
        commit_message = row.get("commit_message") or ""
        diff = row.get("diff") or ""
        reflete_mudanca = row.get("reflete_mudanca") or row.get("reflete_mudanca_canon") or ""
        qualidade = row.get("qualidade")

        if not repo or not commit_hash or not diff:
            log(f"Skip row {idx}: missing repo, commit_hash, or diff")
            continue

        key = _make_key(repo, commit_hash)
        if key in existing_keys:
            log(f"Skip row {idx}: already in output ({key})")
            continue

        sent += 1
        log(f"Calling Claude {sent}/{args.limit if args.limit else '∞'} for {key}")

        prompt = build_prompt(
            diff=diff,
            commit_message=commit_message,
            reflete_mudanca=str(reflete_mudanca),
            qualidade=qualidade,
        )

        text = call_claude(api_key=api_key, prompt=prompt)

        reflete_just, qualidade_just = parse_justification_response(text)

        results.append(
            {
                "repo": repo,
                "commit_hash": commit_hash,
                "reflete_mudanca": row.get("reflete_mudanca"),
                "reflete_mudanca_canon": row.get("reflete_mudanca_canon"),
                "reflete_mudanca_justificativa": reflete_just,
                "qualidade": qualidade,
                "qualidade_justificativa": qualidade_just,
                "commit_message": commit_message,
                "diff": diff,
                "raw": text,
            }
        )
        existing_keys.add(key)

        # Persist after each request so progress isn't lost.
        write_results(out_path, results)
        log(f"Saved progress: {len(results)} results -> {out_path}")

    log(f"Done. Wrote {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
