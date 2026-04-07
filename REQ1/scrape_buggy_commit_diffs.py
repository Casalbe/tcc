import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

GITHUB_API_BASE = "https://api.github.com"

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


#colunas verificadas e extraidas do CSV de entrada
@dataclass(frozen=True)
class CommitRow:
    commit_hash: str
    contains_bug: bool
    commit_message: str
    la: float
    ld: float


def _parse_bool(value: str) -> bool:
    v = (value or "").strip().lower()
    return v in {"true", "1", "t", "yes", "y"}


def _parse_float(value: str) -> float:
    try:
        return float((value or "").strip())
    except Exception:
        return 0.0


def _normalize_header(name: str) -> str:
    # Handles UTF-8 BOM, stray whitespace, and header casing.
    return (name or "").replace("\ufeff", "").strip().casefold()


def read_rows(csv_path: Path) -> List[CommitRow]:
    required = {"commit_hash", "contains_bug", "commit_message", "la", "ld"}

    # Use utf-8-sig to transparently handle UTF-8 BOM, common in Excel exports.
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(65536)
        f.seek(0)

        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)

        raw_fieldnames = list(reader.fieldnames or [])
        normalized_to_raw: Dict[str, str] = {
            _normalize_header(h): h for h in raw_fieldnames if h is not None
        }

        missing = required - set(normalized_to_raw.keys())
        if missing:
            raise ValueError(
                "CSV missing required columns: "
                f"{sorted(missing)}. "
                f"Detected headers: {raw_fieldnames}"
            )

        rows: List[CommitRow] = []
        for r in reader:
            # Remap row keys to normalized required keys.
            rr = {(_normalize_header(k) if k is not None else ""): v for k, v in r.items()}
            rows.append(
                CommitRow(
                    commit_hash=(rr.get("commit_hash") or "").strip(),
                    contains_bug=_parse_bool(rr.get("contains_bug") or ""),
                    commit_message=(rr.get("commit_message") or "").strip(),
                    la=_parse_float(rr.get("la") or "0"),
                    ld=_parse_float(rr.get("ld") or "0"),
                )
            )
        return rows

#logica de filtragem dos commits (contem bug e possuem churn (la+ld) menor que o valor definido em max_churn (default: 50))
def filter_buggy_small_commits(rows: Iterable[CommitRow], max_churn: float = 50.0) -> List[CommitRow]:
    out: List[CommitRow] = []
    for r in rows:
        if not r.commit_hash:
            continue
        if not r.contains_bug:
            continue
        if (r.la + r.ld) >= max_churn:
            continue
        out.append(r)
    return out


#urls dos commmits
_RE_GH_COMMIT_URL = re.compile(r"^https?://github\\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/commit/(?P<sha>[0-9a-fA-F]{7,40})/?$")
_RE_OWNER_REPO_AT_SHA = re.compile(r"^(?P<owner>[^/]+)/(?P<repo>[^@]+)@(?P<sha>[0-9a-fA-F]{7,40})$")


def parse_commit_locator(locator: str) -> Tuple[str, str, str]:
    """Parse a commit locator into (owner, repo, sha).

    Supported formats:
    - owner/repo@sha
    - https://github.com/owner/repo/commit/sha

    If your CSV only has the SHA, you must pass --repo owner/repo.
    """

    loc = (locator or "").strip()
    m = _RE_OWNER_REPO_AT_SHA.match(loc)
    if m:
        return m.group("owner"), m.group("repo"), m.group("sha")

    m = _RE_GH_COMMIT_URL.match(loc)
    if m:
        return m.group("owner"), m.group("repo"), m.group("sha")

    # Fallback: treat as sha-only
    if re.fullmatch(r"[0-9a-fA-F]{7,40}", loc):
        return "", "", loc

    raise ValueError(
        "Unrecognized commit_hash format. Expected sha, owner/repo@sha, or GitHub commit URL."
    )


def build_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "tcc-req1-scraper",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


#funcao principal de busca dos diffs dos commits
def fetch_commit_diff(owner: str, repo: str, sha: str, headers: Dict[str, str], timeout_s: int = 30) -> str:
    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{sha}"
    diff_headers = dict(headers)
    diff_headers["Accept"] = "application/vnd.github.v3.diff"
    resp = requests.get(url, headers=diff_headers, timeout=timeout_s)
    if resp.status_code == 404:
        raise RuntimeError(f"Commit not found: {owner}/{repo}@{sha}")
    if resp.status_code == 401:
        raise RuntimeError("GitHub API returned 401 Unauthorized. Check your GITHUB_TOKEN.")
    resp.raise_for_status()
    return resp.text


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


def resolve_owner_repo(locator: str, default_repo: Optional[str]) -> Tuple[str, str, str]:
    owner, repo, sha = parse_commit_locator(locator)
    if owner and repo:
        return owner, repo, sha

    if not default_repo or "/" not in default_repo:
        raise ValueError(
            "CSV commit_hash looks like a SHA only. Provide --repo owner/repo to locate commits."
        )
    default_owner, default_repo_name = default_repo.split("/", 1)
    return default_owner, default_repo_name, sha


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch diffs for buggy low-churn commits listed in a CSV.")
    p.add_argument("--input", "-i", required=True, help="Path to input CSV")
    p.add_argument("--output", "-o", required=True, help="Path to output JSON")
    p.add_argument(
        "--repo",
        help="Default repo in owner/repo form (required when commit_hash is SHA only)",
        default="belaban/JGroups",
    )
    p.add_argument(
        "--max-churn",
        type=float,
        default=50.0,
        help="Only include commits where la+ld < max-churn (default: 50)",
    )
    p.add_argument(
        "--token",
        help="GitHub token (default: env GITHUB_TOKEN)",
        default=None,
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Stop after this many successful commit objects are written to the output (default: 100)",
    )

    args = p.parse_args()

    # carregar .env se existir para obter o token do github (evitando expor o token em variaveis de ambiente ou CLI history)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    load_dotenv(repo_root / ".env")

    in_path = Path(args.input)
    out_path = Path(args.output)

    rows = read_rows(in_path)
    selected = filter_buggy_small_commits(rows, max_churn=args.max_churn)

    log(f"Loaded {len(rows)} CSV rows")
    log(f"Selected {len(selected)} rows where contains_bug=true and la+ld < {args.max_churn}")
    log(f"Will stop after {args.limit} successful diffs")

    token = args.token or os.getenv("GITHUB_TOKEN")
    if token:
        log("GitHub token detected (authenticated requests)")
    else:
        log("No GitHub token found; you may hit low rate limits")
    headers = build_headers(token)

    results: List[Dict[str, Any]] = []
    attempted = 0
    for r in selected:
        if len(results) >= args.limit:
            break

        attempted += 1
        try:
            owner, repo, sha = resolve_owner_repo(r.commit_hash, args.repo)
            log(f"[{len(results)}/{args.limit}] Fetching diff {attempted}/{len(selected)}: {owner}/{repo}@{sha}")
            diff = fetch_commit_diff(owner, repo, sha, headers=headers)
        except Exception as e:
            log(f"Skip (error): {r.commit_hash} -> {e}")
            continue

        results.append({"commit_message": r.commit_message, "diff": diff})

        # mantem o hash do commit pois a analise posterior necessita
        results[-1]["commit_hash"] = sha

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(f"Done. Wrote {len(results)} objects to {out_path}")


if __name__ == "__main__":
    main()
