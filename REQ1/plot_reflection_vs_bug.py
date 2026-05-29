import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from REQ1.train_reflection_models import LABELS_ORDERED


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _order_categories(present: Dict[str, int]) -> List[str]:
    ordered: List[str] = [c for c in LABELS_ORDERED if int(present.get(c, 0) or 0) > 0]
    # If something unexpected appears, append it deterministically
    extras = sorted([k for k in present.keys() if k not in LABELS_ORDERED and int(present[k] or 0) > 0])
    return ordered + extras


def _stacked_two_bars(
    ax: Any,
    *,
    title: str,
    categories: List[str],
    clean_counts: Dict[str, int],
    bug_counts: Dict[str, int],
    normalize: bool,
) -> None:
    labels = ["clean", "defect_inducing"]
    totals = [sum(int(clean_counts.get(c, 0) or 0) for c in categories), sum(int(bug_counts.get(c, 0) or 0) for c in categories)]

    x = np.array([0, 1], dtype=float)
    bottoms = np.array([0.0, 0.0], dtype=float)

    for cat in categories:
        raw = np.array([
            float(clean_counts.get(cat, 0) or 0),
            float(bug_counts.get(cat, 0) or 0),
        ])
        if normalize:
            vals = np.array([
                (100.0 * raw[0] / totals[0]) if totals[0] else 0.0,
                (100.0 * raw[1] / totals[1]) if totals[1] else 0.0,
            ])
        else:
            vals = raw

        ax.bar(x, vals, bottom=bottoms, label=cat)
        bottoms = bottoms + vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_ylabel("Percentual (%)" if normalize else "Contagem")
    if normalize:
        ax.set_ylim(0, 100)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Create stacked bar chart comparing clean vs defect-inducing commits from reflection_vs_bug.json"
    )
    p.add_argument(
        "--in",
        dest="in_path",
        default=str(Path("REQ1") / "analysis_out" / "reflection_vs_bug.json"),
        help="Input JSON report (default: REQ1/analysis_out/reflection_vs_bug.json)",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        default=str(Path("REQ1") / "analysis_out" / "reflection_vs_bug_stacked.png"),
        help="Output image path (default: REQ1/analysis_out/reflection_vs_bug_stacked.png)",
    )
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Plot raw counts instead of percentages",
    )

    args = p.parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    normalize = not args.no_normalize

    data = _read_json(in_path)
    splits: Dict[str, Any] = data.get("splits") or {}
    if not splits:
        raise SystemExit(f"No 'splits' found in {in_path}")

    # Determine categories that appear in any split
    present: Dict[str, int] = {}
    for part in ("train", "test"):
        block = splits.get(part) or {}
        for key in ("distribution_clean", "distribution_bug"):
            dist = block.get(key) or {}
            for k, v in dist.items():
                present[k] = present.get(k, 0) + int(v or 0)

    categories = _order_categories(present)
    if not categories:
        raise SystemExit("No reflection categories with non-zero counts to plot")

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from e

    # Create 1x2 layout (train/test)
    parts: List[Tuple[str, str]] = []
    if "train" in splits:
        parts.append(("train", "Train"))
    if "test" in splits:
        parts.append(("test", "Test"))

    fig, axes = plt.subplots(1, len(parts), figsize=(max(9, 6 * len(parts)), 6), squeeze=False)

    for idx, (part_name, part_title) in enumerate(parts):
        ax = axes[0][idx]
        block = splits.get(part_name) or {}
        clean = block.get("distribution_clean") or {}
        bug = block.get("distribution_bug") or {}
        _stacked_two_bars(
            ax,
            title=f"{part_title}: clean vs defect-inducing",
            categories=categories,
            clean_counts=clean,
            bug_counts=bug,
            normalize=normalize,
        )

    # Put legend outside to avoid clutter
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote stacked bar chart to {out_path}")


if __name__ == "__main__":
    main()
