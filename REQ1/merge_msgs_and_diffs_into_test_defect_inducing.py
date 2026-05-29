import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _repo_from_filename(path: Path) -> str:
    name = path.stem  # msgs_and_diffs_camel
    suffix = name.split("msgs_and_diffs_", 1)[-1]

    # Map dataset repo-key to GitHub slug used elsewhere in REQ1.
    mapping = {
        "camel": "ceejay66/camel",
        "fabric8": "jboss-fuse/fabric8",
        "gimp": "bolabola/gimp",
        "neutron": "subramani95/neutron",
        "postgresql": "krhancoc/postgres",
        "jgroups": "belaban/JGroups",
    }
    return mapping.get(suffix, suffix)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _commit_hash(item: Dict[str, Any]) -> str:
    return str(item.get("commit_hash") or "").strip()


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Merge existing msgs_and_diffs_*.json files into a single test defect-inducing split JSON, "
            "with de-duplication by commit_hash."
        )
    )
    p.add_argument(
        "--msgs-and-diffs-dir",
        default=str(Path(__file__).resolve().parent / "data" / "msgs and diffs"),
        help="Folder containing msgs_and_diffs_*.json (default: REQ1/data/msgs and diffs)",
    )
    p.add_argument(
        "--test-clean",
        default=str(Path(__file__).resolve().parent / "splits" / "test" / "clean.json"),
        help="Path to test clean JSON to check for overlap (default: REQ1/splits/test/clean.json)",
    )
    p.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "splits" / "test" / "defect_inducing.json"),
        help="Output path for merged test defect-inducing JSON (default: REQ1/splits/test/defect_inducing.json)",
    )
    p.add_argument(
        "--allow-overlap-with-test-clean",
        action="store_true",
        help="If set, do not error when a commit_hash overlaps with test clean; overlapping items are dropped.",
    )

    args = p.parse_args()

    msgs_dir = Path(args.msgs_and_diffs_dir)
    out_path = Path(args.output)
    test_clean_path = Path(args.test_clean)

    if not msgs_dir.exists():
        raise FileNotFoundError(f"msgs-and-diffs dir not found: {msgs_dir}")

    files = sorted(msgs_dir.glob("msgs_and_diffs_*.json"))
    if not files:
        raise FileNotFoundError(f"No msgs_and_diffs_*.json files found in: {msgs_dir}")

    test_clean_hashes: Set[str] = set()
    if test_clean_path.exists():
        clean = _load_json(test_clean_path)
        if isinstance(clean, list):
            for item in clean:
                if isinstance(item, dict):
                    sha = _commit_hash(item)
                    if sha:
                        test_clean_hashes.add(sha)

    merged: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    dropped_duplicates = 0
    dropped_overlap = 0

    for f in files:
        data = _load_json(f)
        if not isinstance(data, list):
            continue

        repo = _repo_from_filename(f)

        for item in data:
            if not isinstance(item, dict):
                continue
            sha = _commit_hash(item)
            if not sha:
                continue

            if sha in test_clean_hashes:
                dropped_overlap += 1
                if not args.allow_overlap_with_test_clean:
                    raise RuntimeError(
                        f"Overlap detected with test clean: {sha} (from {f.name}). "
                        "Rebuild test clean or set --allow-overlap-with-test-clean to drop overlaps."
                    )
                continue

            if sha in seen:
                dropped_duplicates += 1
                continue

            seen.add(sha)
            merged.append(
                {
                    "commit_hash": sha,
                    "repo": repo,
                    "commit_message": str(item.get("commit_message") or ""),
                    "diff": str(item.get("diff") or ""),
                    "contains_bug": True,
                    "split": "test",
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Merged {len(merged)} items into {out_path}. "
        f"Dropped duplicates={dropped_duplicates}, dropped overlap with test clean={dropped_overlap}."
    )


if __name__ == "__main__":
    main()
