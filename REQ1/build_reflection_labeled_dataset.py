import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


LABEL_FIELD = "reflete_mudanca"
HASH_FIELD = "commit_hash"

ALLOWED_LABELS = {"Sim", "Provavelmente sim", "Provavelmente não", "Não"}


@dataclass(frozen=True)
class LabeledCommit:
    commit_hash: str
    project: str
    commit_message: str
    diff: str
    reflete_mudanca: str
    qualidade: Optional[int]


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {path}")
    return data


def load_labels(claude_results_paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    by_hash: Dict[str, Dict[str, Any]] = {}

    for p in claude_results_paths:
        items = _read_json_list(p)
        for obj in items:
            if not isinstance(obj, dict):
                continue
            h = (obj.get(HASH_FIELD) or "").strip()
            label = (obj.get(LABEL_FIELD) or "").strip()
            if not h or not label:
                continue
            by_hash[h] = obj

    return by_hash


def iter_commits(msgs_and_diffs_paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for p in msgs_and_diffs_paths:
        items = _read_json_list(p)
        for obj in items:
            if isinstance(obj, dict):
                obj["_source_file"] = p.name
                yield obj


def infer_project_from_filename(filename: str) -> str:
    # msgs_and_diffs_<project>.json
    stem = Path(filename).stem
    if stem.startswith("msgs_and_diffs_"):
        return stem[len("msgs_and_diffs_") :]
    return stem


def build_dataset(
    labels_by_hash: Dict[str, Dict[str, Any]],
    commit_iter: Iterable[Dict[str, Any]],
) -> List[LabeledCommit]:
    out: List[LabeledCommit] = []

    for obj in commit_iter:
        h = (obj.get(HASH_FIELD) or "").strip()
        if not h:
            continue
        label_obj = labels_by_hash.get(h)
        if not label_obj:
            continue

        commit_message = obj.get("commit_message") or ""
        diff = obj.get("diff") or ""
        if not commit_message or not diff:
            continue

        reflete = (label_obj.get(LABEL_FIELD) or "").strip()
        if reflete not in ALLOWED_LABELS:
            # The upstream LLM prompt allowed values outside the 4-class thesis scale (e.g., "Sem mensagem").
            # We exclude them here to keep the dataset aligned with the thesis categories.
            continue
        qualidade = label_obj.get("qualidade")
        try:
            qualidade_int = int(qualidade) if qualidade is not None else None
        except Exception:
            qualidade_int = None

        project = infer_project_from_filename(obj.get("_source_file") or "")

        out.append(
            LabeledCommit(
                commit_hash=h,
                project=project,
                commit_message=commit_message,
                diff=diff,
                reflete_mudanca=reflete,
                qualidade=qualidade_int,
            )
        )

    return out


def write_jsonl(path: Path, rows: List[LabeledCommit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "commit_hash": r.commit_hash,
                        "project": r.project,
                        "commit_message": r.commit_message,
                        "diff": r.diff,
                        "reflete_mudanca": r.reflete_mudanca,
                        "qualidade": r.qualidade,
                    },
                    ensure_ascii=False,
                )
            )
            f.write("\n")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Join Claude commit-message reflection labels (claude_results_*.json) "
            "with scraped message+diff data (msgs_and_diffs_*.json) and produce a JSONL dataset."
        )
    )
    p.add_argument(
        "--claude-results-dir",
        default=str(Path("REQ1") / "data" / "claude results"),
        help="Directory containing claude_results_*.json",
    )
    p.add_argument(
        "--msgs-and-diffs-dir",
        default=str(Path("REQ1") / "data" / "msgs and diffs"),
        help="Directory containing msgs_and_diffs_*.json",
    )
    p.add_argument(
        "--out",
        default=str(Path("REQ1") / "data" / "reflection_labeled_commits.jsonl"),
        help="Output JSONL path",
    )

    args = p.parse_args()

    claude_dir = Path(args.claude_results_dir)
    msgs_dir = Path(args.msgs_and_diffs_dir)
    out_path = Path(args.out)

    claude_paths = sorted(claude_dir.glob("claude_results_*.json"))
    msgs_paths = sorted(msgs_dir.glob("msgs_and_diffs_*.json"))

    if not claude_paths:
        raise SystemExit(f"No claude_results_*.json found in {claude_dir}")
    if not msgs_paths:
        raise SystemExit(f"No msgs_and_diffs_*.json found in {msgs_dir}")

    labels_by_hash = load_labels(claude_paths)
    rows = build_dataset(labels_by_hash, iter_commits(msgs_paths))

    # stable output order
    rows.sort(key=lambda r: (r.project, r.commit_hash))

    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} labeled commits to {out_path}")
    print(f"Allowed labels: {', '.join(sorted(ALLOWED_LABELS))}")


if __name__ == "__main__":
    main()
