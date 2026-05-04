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
    quality_values: List[int]
    quality_values_by_reflection: Dict[str, List[int]]
    no_message_count: int
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


def plot_quality_boxplot(
    data: List[Tuple[str, List[int], Optional[int]]],
    title: str,
    out_path: Path,
) -> None:
    """Boxplot of quality scores.

    Uses standard quartiles (25/50/75) for the box and sets whiskers to the
    0th and 100th percentiles (min/max) to match the professor's guidance.

    `data` items: (label, quality_values, no_message_count)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    filtered = [(label, values, no_msg) for label, values, no_msg in data if values]
    if not filtered:
        raise ValueError("No non-empty commit messages with quality scores to plot")

    labels: List[str] = []
    for label, values, no_msg in filtered:
        if no_msg is None:
            labels.append(f"{label}\n(n={len(values)})")
        else:
            labels.append(f"{label}\n(n={len(values)}, sem msg={no_msg})")
    series = [values for _, values, _ in filtered]

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(series)), 6))
    ax.boxplot(
        series,
        labels=labels,
        whis=(0, 100),
        showfliers=False,
        showmeans=True,
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Qualidade (0–100)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_reflection_stacked_by_repo(
    repo_results: List[RepoResult],
    title: str,
    out_path: Path,
    *,
    normalize: bool = True,
) -> None:
    """Stacked columns: one bar per repo, stacked by reflection class.

    If normalize=True, each bar sums to 100% (recommended for comparing tendencies
    when repos have different sample sizes).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    if not repo_results:
        raise ValueError("No repositories to plot")

    repo_labels = [r.name for r in repo_results]
    totals = [sum(int(v) for v in r.reflection_counts.values()) for r in repo_results]

    # Only include categories that actually appear in the dataset
    categories = [
        cat
        for cat in _REFLECTION_ORDER
        if any((r.reflection_counts.get(cat, 0) or 0) > 0 for r in repo_results)
    ]
    if not categories:
        raise ValueError("No reflection categories found")

    x = list(range(len(repo_results)))
    bottoms = [0.0 for _ in repo_results]

    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(repo_results)), 6))

    for cat in categories:
        values: List[float] = []
        for rr, total in zip(repo_results, totals):
            count = float(rr.reflection_counts.get(cat, 0) or 0)
            if normalize:
                values.append((100.0 * count / total) if total else 0.0)
            else:
                values.append(count)

        ax.bar(x, values, bottom=bottoms, label=cat)
        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_xticks(x)
    ax.set_xticklabels(repo_labels, rotation=20, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_ylabel("Percentual (%)" if normalize else "Contagem")
    if normalize:
        ax.set_ylim(0, 100)

    # Put legend outside to avoid clutter
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
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
    quality_values: List[int] = []
    quality_values_by_reflection: Dict[str, List[int]] = {k: [] for k in _REFLECTION_ORDER}
    no_message_count = 0

    for item in raw_items:
        commit_hash = str(item.get("commit_hash") or "").strip()
        reflection = normalize_reflection(item.get("reflete_mudanca"))
        quality = normalize_quality(item.get("qualidade"))

        # Decide "no message" using msgs_and_diffs if available; otherwise fall back to Claude label.
        if commit_hash and commit_hash in commit_messages:
            is_no_message = looks_like_no_message(commit_messages[commit_hash])
        else:
            is_no_message = reflection == "Sem mensagem"

        if is_no_message:
            no_message_count += 1
        else:
            if quality is not None:
                quality_values.append(quality)

        quality_bucket = quality_bin(quality, is_no_message=is_no_message)
        quality_counts[quality_bucket] += 1

        reflection_bucket = "Sem mensagem" if is_no_message else reflection
        if reflection_bucket not in _REFLECTION_ORDER:
            reflection_bucket = "Outro"
        reflection_counts[reflection_bucket] += 1

        # For correlation plots: quality distribution grouped by reflection class.
        # Exclude empty-message commits from the numeric distribution.
        if (not is_no_message) and (quality is not None):
            bucket_for_box = reflection_bucket
            if bucket_for_box in {"Sem mensagem"}:
                bucket_for_box = "Outro"
            if bucket_for_box not in quality_values_by_reflection:
                quality_values_by_reflection[bucket_for_box] = []
            quality_values_by_reflection[bucket_for_box].append(quality)

    name = results_path.stem
    if name.startswith("claude_results_"):
        name = name[len("claude_results_") :]

    return RepoResult(
        name=name,
        quality_counts=quality_counts,
        reflection_counts=reflection_counts,
        quality_values=quality_values,
        quality_values_by_reflection=quality_values_by_reflection,
        no_message_count=no_message_count,
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
    all_quality_values: List[int] = []
    all_no_message_count = 0
    all_quality_values_by_reflection: Dict[str, List[int]] = {k: [] for k in _REFLECTION_ORDER}

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
        all_quality_values.extend(rr.quality_values)
        all_no_message_count += rr.no_message_count
        for k, vals in rr.quality_values_by_reflection.items():
            if not vals:
                continue
            if k not in all_quality_values_by_reflection:
                all_quality_values_by_reflection[k] = []
            all_quality_values_by_reflection[k].extend(vals)

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

        # Boxplot for this repo's quality distribution (excluding empty messages)
        if rr.quality_values:
            plot_quality_boxplot(
                [(rr.name, rr.quality_values, rr.no_message_count)],
                title=f"{rr.name}: Distribuição da qualidade (boxplot)\nWhiskers=0–100 percentis; Box=25–75; Mediana=50",
                out_path=out_dir / f"{rr.name}_quality_boxplot.png",
            )

        # Boxplot grouped by reflection class (best for seeing correlation)
        box_groups = []
        for key in _REFLECTION_ORDER:
            if key in {"Sem mensagem"}:
                continue
            values = rr.quality_values_by_reflection.get(key) or []
            box_groups.append((key, values, None))
        # Include "Outro" if present
        if rr.quality_values_by_reflection.get("Outro"):
            box_groups.append(("Outro", rr.quality_values_by_reflection["Outro"], None))

        try:
            plot_quality_boxplot(
                box_groups,
                title=(
                    f"{rr.name}: Qualidade por classe de reflexão (boxplot)\n"
                    f"(exclui 'Sem mensagem' do boxplot; sem msg={rr.no_message_count})\n"
                    "Whiskers=0–100 percentis; Box=25–75; Mediana=50"
                ),
                out_path=out_dir / f"{rr.name}_quality_boxplot_by_reflection.png",
            )
        except ValueError:
            # No non-empty quality values to plot for this repo
            pass

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

        # Combined boxplot: side-by-side per repo (quality only, excluding empty messages)
        per_repo_box_data = [(r.name, r.quality_values, r.no_message_count) for r in repo_results]
        try:
            plot_quality_boxplot(
                per_repo_box_data,
                title=(
                    "Todos os repositórios: Distribuição da qualidade (boxplot)\n"
                    "Whiskers=0–100 percentis; Box=25–75; Mediana=50"
                ),
                out_path=out_dir / "all_quality_boxplot_by_repo.png",
            )
        except ValueError:
            # If some repos have no data, plot_quality_boxplot filters empties; if all empty, skip.
            pass

        # Combined boxplot: one distribution across all repos
        if all_quality_values:
            plot_quality_boxplot(
                [("all", all_quality_values, all_no_message_count)],
                title=(
                    "Todos os repositórios: Qualidade (boxplot agregado)\n"
                    "Whiskers=0–100 percentis; Box=25–75; Mediana=50"
                ),
                out_path=out_dir / "all_quality_boxplot.png",
            )

        # Combined boxplot grouped by reflection class
        combined_groups: List[Tuple[str, List[int], Optional[int]]] = []
        for key in _REFLECTION_ORDER:
            if key in {"Sem mensagem"}:
                continue
            combined_groups.append((key, all_quality_values_by_reflection.get(key, []), None))
        if all_quality_values_by_reflection.get("Outro"):
            combined_groups.append(("Outro", all_quality_values_by_reflection["Outro"], None))

        try:
            plot_quality_boxplot(
                combined_groups,
                title=(
                    "Todos os repositórios: Qualidade por classe de reflexão (boxplot)\n"
                    f"(exclui 'Sem mensagem' do boxplot; sem msg={all_no_message_count})\n"
                    "Whiskers=0–100 percentis; Box=25–75; Mediana=50"
                ),
                out_path=out_dir / "all_quality_boxplot_by_reflection.png",
            )
        except ValueError:
            pass

        # Stacked chart: one column per repo, layers are reflection categories
        try:
            plot_reflection_stacked_by_repo(
                repo_results,
                title=(
                    "Reflexão da mudança por repositório (100% stacked)\n"
                    "Cada coluna = um repositório; camadas = classes de reflexão"
                ),
                out_path=out_dir / "reflection_stacked_by_repo.png",
                normalize=True,
            )
        except ValueError:
            pass

    print(f"Wrote outputs to: {out_dir}")
    for rr in repo_results:
        print(f"- {rr.name}: n={rr.total} (sem mensagem={rr.no_message_count})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
