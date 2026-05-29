import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
from scipy.stats import chi2_contingency

from REQ1.train_reflection_models import (
    LABELS_ORDERED,
    FeatureBundle,
    build_features,
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_split(path: Path) -> List[Dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")
    return data


def predict_labels(
    rows: List[Dict[str, Any]],
    bundle: FeatureBundle,
    model: Any,
    max_diff_chars: int,
) -> np.ndarray:
    # Adapt the split rows to the feature builder expected schema
    adapted = []
    for r in rows:
        adapted.append(
            {
                "commit_message": r.get("commit_message") or "",
                "diff": r.get("diff") or "",
                # dummy label required by build_features; won't be used
                "reflete_mudanca": "Sim",
            }
        )
    X, _, _ = build_features(
        adapted,
        vec_msg=bundle.vec_msg,
        vec_diff=bundle.vec_diff,
        fit=False,
        max_diff_chars=max_diff_chars,
    )
    return model.predict(X)


def to_label_name(ids: np.ndarray) -> List[str]:
    return [LABELS_ORDERED[int(i)] for i in ids.tolist()]


def contingency_table(pred_labels: List[str], contains_bug: List[bool]) -> Tuple[np.ndarray, List[str]]:
    label_to_idx = {lab: i for i, lab in enumerate(LABELS_ORDERED)}
    table = np.zeros((2, len(LABELS_ORDERED)), dtype=np.int64)
    for lab, bug in zip(pred_labels, contains_bug):
        row = 1 if bug else 0
        col = label_to_idx.get(lab, None)
        if col is None:
            continue
        table[row, col] += 1
    return table, LABELS_ORDERED


def drop_all_zero_columns(table: np.ndarray, cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    col_sums = table.sum(axis=0)
    keep = col_sums > 0
    filtered = table[:, keep]
    filtered_cols = [c for c, k in zip(cols, keep.tolist()) if k]
    return filtered, filtered_cols


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Apply a trained reflection classifier to defect-inducing vs clean splits, "
            "and compute label distributions + chi-square association test."
        )
    )
    p.add_argument(
        "--model-dir",
        default=str(Path("REQ1") / "models" / "reflection"),
        help="Directory produced by train_reflection_models.py",
    )
    p.add_argument(
        "--model",
        choices=["logreg", "linear_svm", "xgboost"],
        default="linear_svm",
        help="Which trained model to use",
    )
    p.add_argument(
        "--splits-dir",
        default=str(Path("REQ1") / "splits"),
        help="Directory containing train/test clean/defect_inducing JSON files",
    )
    p.add_argument(
        "--out",
        default=str(Path("REQ1") / "analysis_out" / "reflection_vs_bug.json"),
        help="Output JSON report path",
    )
    p.add_argument("--max-diff-chars", type=int, default=6000)

    args = p.parse_args()

    model_dir = Path(args.model_dir)
    bundle: FeatureBundle = joblib.load(model_dir / "feature_bundle.joblib")
    model = joblib.load(model_dir / f"{args.model}.joblib")

    splits_dir = Path(args.splits_dir)
    partitions = {
        "train": {
            "clean": splits_dir / "train" / "clean.json",
            "defect_inducing": splits_dir / "train" / "defect_inducing.json",
        },
        "test": {
            "clean": splits_dir / "test" / "clean.json",
            "defect_inducing": splits_dir / "test" / "defect_inducing.json",
        },
    }

    for part_name, group in partitions.items():
        for group_name, pth in group.items():
            if not pth.exists():
                raise SystemExit(f"Missing split file: {part_name}/{group_name} -> {pth}")

    report: Dict[str, Any] = {
        "model": args.model,
        "labels": LABELS_ORDERED,
        "splits": {},
    }

    for part_name, group in partitions.items():
        rows_clean = load_split(group["clean"])
        rows_bug = load_split(group["defect_inducing"])

        preds_clean = to_label_name(predict_labels(rows_clean, bundle, model, args.max_diff_chars))
        preds_bug = to_label_name(predict_labels(rows_bug, bundle, model, args.max_diff_chars))

        pred_names = preds_clean + preds_bug
        bugs = ([False] * len(preds_clean)) + ([True] * len(preds_bug))

        dist_total = Counter(pred_names)
        dist_bug = Counter(preds_bug)
        dist_clean = Counter(preds_clean)

        table, cols = contingency_table(pred_names, bugs)
        table_chi2, cols_chi2 = drop_all_zero_columns(table, cols)

        chi_square: Dict[str, Any]
        if table_chi2.shape[1] < 2:
            chi_square = {
                "skipped": True,
                "reason": "Need at least 2 predicted label categories to run chi-square; some labels had zero counts.",
            }
        else:
            chi2, p_value, dof, expected = chi2_contingency(table_chi2)
            chi_square = {
                "chi2": float(chi2),
                "p_value": float(p_value),
                "dof": int(dof),
                "expected": expected.tolist(),
            }

        report["splits"][part_name] = {
            "n": len(pred_names),
            "n_clean": len(preds_clean),
            "n_defect_inducing": len(preds_bug),
            "distribution_total": dict(dist_total),
            "distribution_bug": dict(dist_bug),
            "distribution_clean": dict(dist_clean),
            "contingency_table": {
                "rows": ["clean", "defect_inducing"],
                "cols": cols,
                "values": table.tolist(),
            },
            "chi_square": chi_square,
            "chi_square_input": {
                "cols": cols_chi2,
                "values": table_chi2.tolist(),
            },
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
