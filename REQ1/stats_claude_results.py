from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


_REFLECTION_ORDER: Tuple[str, ...] = (
    "Sim",
    "Provavelmente sim",
    "Provavelmente não",
    "Não",
    "Sem mensagem",
    "Outro",
)

_QUALITY_ORDER: Tuple[str, ...] = (
    "1-25",
    "26-50",
    "51-75",
    "76-100",
    "Sem mensagem",
    "Outro",
)

_EMPTY_MESSAGE_MARKERS = {
    "",
    "*** empty log message ***",
    "no message",
    "(no message)",
    "[no message]",
    "sem mensagem",
}

_EMPTY_MESSAGE_MARKERS_NORM = {m.strip().lower() for m in _EMPTY_MESSAGE_MARKERS}


def _strip_accents(text: str) -> str:
    # NFKD + drop combining marks
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def normalize_reflection(label: Optional[str]) -> str:
    if label is None:
        return "Outro"

    s = _normalize_spaces(str(label))

    # Some stored labels include extra quotes: '"Provavelmente sim"'
    # Strip surrounding quotes repeatedly.
    while (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in {"\"", "'"}):
        s = _normalize_spaces(s[1:-1])

    key = _strip_accents(s).lower()

    mapping = {
        "sim": "Sim",
        "nao": "Não",
        "não": "Não",
        "provavelmente sim": "Provavelmente sim",
        "provavelmente nao": "Provavelmente não",
        "provavelmente não": "Provavelmente não",
        "sem mensagem": "Sem mensagem",
    }

    return mapping.get(key, s if s else "Outro")


def normalize_quality(value: Any) -> Optional[int]:
    if value is None:
        return None

    if isinstance(value, bool):
        return int(value)

    try:
        q = int(str(value).strip())
    except Exception:
        return None

    if q < 0:
        q = 0
    if q > 100:
        q = 100
    return q


def quality_bin(quality: Optional[int], *, is_no_message: bool) -> str:
    if is_no_message:
        return "Sem mensagem"

    if quality is None:
        return "Outro"

    # Treat 0 as the lowest bin (rare unless Sem mensagem).
    if quality <= 25:
        return "1-25"
    if quality <= 50:
        return "26-50"
    if quality <= 75:
        return "51-75"
    return "76-100"


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, got: {type(data).__name__}")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Expected list[dict] items in {path}, item {i} is {type(item).__name__}"
            )
        out.append(item)
    return out


def load_commit_messages_from_msgs_and_diffs(path: Path) -> Dict[str, str]:
    items = load_json_list(path)
    mapping: Dict[str, str] = {}
    for item in items:
        commit_hash = str(item.get("commit_hash") or "").strip()
        if not commit_hash:
            continue
        commit_message = item.get("commit_message")
        mapping[commit_hash] = "" if commit_message is None else str(commit_message)
    return mapping


def looks_like_no_message(commit_message: Optional[str]) -> bool:
    if commit_message is None:
        return True
    msg = _normalize_spaces(str(commit_message))

    lowered = msg.lower()
    if lowered in _EMPTY_MESSAGE_MARKERS_NORM:
        return True

    # Handle common placeholder variants with punctuation/extra whitespace
    # Examples: "no message.", "No message!", "*** empty log message ***"
    lowered_clean = re.sub(r"[^a-z*\s]", "", lowered).strip()
    if lowered_clean in _EMPTY_MESSAGE_MARKERS_NORM:
        return True

    # Very defensive: if the entire message is just a placeholder sentence.
    if re.fullmatch(r"(?:no|sem)\s+mensagem", lowered_clean):
        return True

    return False


@dataclass(frozen=True)
class RepoResult:
    name: str
    quality_counts: Counter[str]
    reflection_counts: Counter[str]
    total: int


def _order_counts(counts: Mapping[str, int], order: Tuple[str, ...]) -> List[Tuple[str, int]]:
    present = dict(counts)
    ordered: List[Tuple[str, int]] = []
    for key in order:
        if key in present:
            ordered.append((key, int(present.pop(key))))
    for key in sorted(present.keys()):
        ordered.append((key, int(present[key])))
    return ordered


def plot_pie(counts: Mapping[str, int], title: str, out_path: Path, order: Tuple[str, ...]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    ordered = [(k, v) for k, v in _order_counts(counts, order) if v > 0]
    if not ordered:
        raise ValueError("No data to plot")

    labels = [f"{k} ({v})" for k, v in ordered]
    values = [v for _, v in ordered]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(values, labels=None, startangle=90, autopct="%1.1f%%")
    ax.axis("equal")
    ax.set_title(title)
    ax.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def analyze_repo(
    results_path: Path,
    *,
    msgs_and_diffs_path: Optional[Path],
) -> RepoResult:
    raw_items = load_json_list(results_path)

    commit_messages: Dict[str, str] = {}
    if msgs_and_diffs_path is not None and msgs_and_diffs_path.exists():
        commit_messages = load_commit_messages_from_msgs_and_diffs(msgs_and_diffs_path)

    quality_counts: Counter[str] = Counter()
    reflection_counts: Counter[str] = Counter()

    for item in raw_items:
        commit_hash = str(item.get("commit_hash") or "").strip()
        reflection = normalize_reflection(item.get("reflete_mudanca"))
        quality = normalize_quality(item.get("qualidade"))

        # Decide "no message" using msgs_and_diffs if available; otherwise fall back to Claude label.
        if commit_hash and commit_hash in commit_messages:
            is_no_message = looks_like_no_message(commit_messages[commit_hash])
        else:
            is_no_message = reflection == "Sem mensagem"

        quality_bucket = quality_bin(quality, is_no_message=is_no_message)
        quality_counts[quality_bucket] += 1

        reflection_bucket = "Sem mensagem" if is_no_message else reflection
        if reflection_bucket not in _REFLECTION_ORDER:
            reflection_bucket = "Outro"
        reflection_counts[reflection_bucket] += 1

    name = results_path.stem
    if name.startswith("claude_results_"):
        name = name[len("claude_results_") :]

    return RepoResult(
        name=name,
        quality_counts=quality_counts,
        reflection_counts=reflection_counts,
        total=sum(quality_counts.values()),
    )


def write_counts_csv(counts: Mapping[str, int], out_path: Path, order: Tuple[str, ...]) -> None:
    lines = ["category,count"]
    total = sum(int(v) for v in counts.values())
    for k, v in _order_counts(counts, order):
        lines.append(f"{k},{v}")
    lines.append(f"TOTAL,{total}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate basic statistical plots for claude_results_*.json: "
            "(1) quality-range pie chart + (2) reflection-class pie chart."
        )
    )
    parser.add_argument(
        "--results",
        nargs="*",
        default=None,
        help=(
            "Paths to claude_results_*.json files. If omitted, uses REQ1/claude_results_*.json "
            "relative to this script."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory for images and CSV summaries. Default: REQ1/analysis_out next to this script."
        ),
    )
    parser.add_argument(
        "--no-msgs-and-diffs",
        action="store_true",
        help=(
            "Do not use msgs_and_diffs_*.json to detect 'Sem mensagem'. "
            "If set, detection uses only Claude's reflection label."
        ),
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Do not generate combined (all repos) charts.",
    )

    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    default_out_dir = script_dir / "analysis_out"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out_dir

    if args.results:
        results_paths = [Path(p).resolve() for p in args.results]
    else:
        results_paths = sorted(script_dir.glob("claude_results_*.json"))

    if not results_paths:
        print("No results files found.", file=sys.stderr)
        return 2

    all_quality: Counter[str] = Counter()
    all_reflection: Counter[str] = Counter()

    repo_results: List[RepoResult] = []

    for results_path in results_paths:
        if not results_path.exists():
            print(f"Missing file: {results_path}", file=sys.stderr)
            return 2

        repo = results_path.stem
        if repo.startswith("claude_results_"):
            repo = repo[len("claude_results_") :]

        msgs_path: Optional[Path] = None
        if not args.no_msgs_and_diffs:
            candidate = results_path.with_name(f"msgs_and_diffs_{repo}.json")
            if candidate.exists():
                msgs_path = candidate

        rr = analyze_repo(results_path, msgs_and_diffs_path=msgs_path)
        repo_results.append(rr)

        all_quality.update(rr.quality_counts)
        all_reflection.update(rr.reflection_counts)

        # Plots + CSV
        plot_pie(
            rr.quality_counts,
            title=f"{rr.name}: Qualidade das mensagens (buggy commits)",
            out_path=out_dir / f"{rr.name}_quality_pie.png",
            order=_QUALITY_ORDER,
        )
        plot_pie(
            rr.reflection_counts,
            title=f"{rr.name}: Reflexo da mudança na mensagem (buggy commits)",
            out_path=out_dir / f"{rr.name}_reflection_pie.png",
            order=_REFLECTION_ORDER,
        )
        write_counts_csv(
            rr.quality_counts, out_dir / f"{rr.name}_quality_counts.csv", order=_QUALITY_ORDER
        )
        write_counts_csv(
            rr.reflection_counts,
            out_dir / f"{rr.name}_reflection_counts.csv",
            order=_REFLECTION_ORDER,
        )

    if not args.no_combined and len(repo_results) > 1:
        plot_pie(
            all_quality,
            title="Todos os repositórios: Qualidade das mensagens (buggy commits)",
            out_path=out_dir / "all_quality_pie.png",
            order=_QUALITY_ORDER,
        )
        plot_pie(
            all_reflection,
            title="Todos os repositórios: Reflexo da mudança na mensagem (buggy commits)",
            out_path=out_dir / "all_reflection_pie.png",
            order=_REFLECTION_ORDER,
        )
        write_counts_csv(all_quality, out_dir / "all_quality_counts.csv", order=_QUALITY_ORDER)
        write_counts_csv(
            all_reflection, out_dir / "all_reflection_counts.csv", order=_REFLECTION_ORDER
        )

    print(f"Wrote outputs to: {out_dir}")
    for rr in repo_results:
        print(f"- {rr.name}: n={rr.total}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
