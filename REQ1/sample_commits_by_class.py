import argparse
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPOS = ["camel", "fabric8", "gimp", "neutron", "postgresql"]

# Canonical labels requested by user.
CANONICAL_CLASSES = [
    "sim",
    "provavelmente sim",
    "nao",
    "provavelmente nao",
]


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def normalize_class(label: str) -> Optional[str]:
    """Normalize Claude's `reflete_mudanca` label to one of the canonical classes.

    Returns None if label is unknown.
    """
    if label is None:
        return None

    normalized = _strip_accents(str(label)).strip().lower()
    normalized = " ".join(normalized.split())

    mapping = {
        "sim": "sim",
        "provavelmente sim": "provavelmente sim",
        "nao": "nao",
        "não": "nao",  # just in case accents weren't stripped
        "provavelmente nao": "provavelmente nao",
        "provavelmente não": "provavelmente nao",
    }

    return mapping.get(normalized)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def index_msgs_and_diffs(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_hash: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        commit_hash = rec.get("commit_hash")
        if not commit_hash:
            continue
        by_hash[str(commit_hash)] = rec
    return by_hash


def group_hashes_by_class(
    claude_records: Iterable[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in claude_records:
        commit_hash = rec.get("commit_hash")
        label = rec.get("reflete_mudanca")
        canonical = normalize_class(label)
        if not commit_hash or not canonical:
            continue
        grouped[canonical].append(rec)
    return grouped


def sample_n(
    rng: random.Random,
    items: List[Dict[str, Any]],
    n: int,
    key: str,
) -> List[Dict[str, Any]]:
    """Sample up to n items, deterministically w.r.t. seed + sorted order by `key`."""
    items_sorted = sorted(items, key=lambda x: str(x.get(key, "")))
    if len(items_sorted) <= n:
        return items_sorted
    return rng.sample(items_sorted, n)


def build_samples(
    base_dir: Path,
    repos: List[str],
    per_class: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)

    out_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"seed": seed, "per_class": per_class, "repos": {}}

    for repo in repos:
        msgs_path = base_dir / f"msgs_and_diffs_{repo}.json"
        claude_path = base_dir / f"claude_results_{repo}.json"

        msgs_records = read_json(msgs_path)
        claude_records = read_json(claude_path)

        msgs_by_hash = index_msgs_and_diffs(msgs_records)
        by_class = group_hashes_by_class(claude_records)

        repo_summary: Dict[str, Any] = {
            "available": {},
            "selected": {},
            "missing_msg_or_diff": 0,
        }

        for canonical in CANONICAL_CLASSES:
            candidates = by_class.get(canonical, [])

            # Filter to only those with message+diff available.
            candidates_present: List[Dict[str, Any]] = []
            missing = 0
            for rec in candidates:
                h = str(rec.get("commit_hash"))
                if h in msgs_by_hash:
                    candidates_present.append(rec)
                else:
                    missing += 1

            repo_summary["available"][canonical] = len(candidates_present)
            repo_summary["missing_msg_or_diff"] += missing

            chosen = sample_n(rng, candidates_present, per_class, key="commit_hash")
            repo_summary["selected"][canonical] = len(chosen)

            for rec in chosen:
                h = str(rec["commit_hash"])
                msgdiff = msgs_by_hash[h]
                out_rows.append(
                    {
                        "repo": repo,
                        "commit_hash": h,
                        "reflete_mudanca": rec.get("reflete_mudanca"),
                        "reflete_mudanca_canon": canonical,
                        "qualidade": rec.get("qualidade"),
                        "commit_message": msgdiff.get("commit_message"),
                        "diff": msgdiff.get("diff"),
                    }
                )

        summary["repos"][repo] = repo_summary

    return out_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample N commits per repo per reflete_mudanca class, "
            "and write commit message + diff + classification to JSON."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing msgs_and_diffs_*.json and claude_results_*.json (default: REQ1/) ",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=2,
        help="How many commits to sample per class per repo (default: 2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis_out" / "sampled_commits_by_class.json",
        help="Output JSON file path (default: REQ1/analysis_out/sampled_commits_by_class.json)",
    )
    args = parser.parse_args()

    rows, summary = build_samples(
        base_dir=args.base_dir,
        repos=REPOS,
        per_class=args.per_class,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": rows}
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    total = len(rows)
    print(f"Wrote {total} sampled commits -> {args.out}")
    for repo, s in summary["repos"].items():
        selected = s["selected"]
        available = s["available"]
        print(
            f"{repo}: selected="
            + ", ".join(f"{k}={selected.get(k, 0)}" for k in CANONICAL_CLASSES)
            + " | available="
            + ", ".join(f"{k}={available.get(k, 0)}" for k in CANONICAL_CLASSES)
            + (f" | missing_msg_or_diff={s['missing_msg_or_diff']}" if s["missing_msg_or_diff"] else "")
        )


if __name__ == "__main__":
    main()
