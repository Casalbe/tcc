"""Train a multiclass Logistic Regression classifier to predict Claude's reflection label.

Data sources (workspace-local):
- Features: REQ1/data/csvs/*_rot_gpt.csv (semicolon-delimited)
- Labels:   REQ1/data/claude results/claude_results_*.json (commit_hash -> reflete_mudanca)
- Splits:   REQ1/splits/{train,test}/defect_inducing.json

Outputs:
- A simplified labeled CSV without the 2 contains_bug columns and 2 vl columns.
- A metrics CSV with test errors stratified by claude_ans + recall.

The classifier target has exactly 4 classes:
  Provavelmente sim, Sim, Provavelmente não, Não
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


CLASS_ORDER: Tuple[str, ...] = (
    "Provavelmente sim",
    "Sim",
    "Provavelmente não",
    "Não",
)

DROP_COLUMNS: Tuple[str, ...] = (
    "contains_bug",
    "contains_bug_gpt",
    "vl",
    "vl_gpt",
)


def load_reflection_jsonl_dataset(jsonl_path: Path) -> pd.DataFrame:
    """Load labeled commits from reflection_labeled_commits.jsonl.

    Expected fields (per line):
      - commit_hash, project, commit_message, diff, reflete_mudanca, (optional) qualidade
    """

    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL dataset not found: {jsonl_path}")

    rows: List[Dict[str, Any]] = []
    bad_lines = 0
    for raw_line in jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            bad_lines += 1
            continue

        commit_hash = str(obj.get("commit_hash") or "").strip()
        project = normalize_project_key(obj.get("project"))
        commit_message = str(obj.get("commit_message") or "").strip()
        diff = str(obj.get("diff") or "").strip()
        label = normalize_claude_label(obj.get("reflete_mudanca"))
        if not commit_hash or not project or not label:
            continue

        row: Dict[str, Any] = {
            "commit_hash": commit_hash,
            "project": project,
            "commit_message": commit_message,
            "diff": diff,
            "claude_ans": label,
        }
        if "qualidade" in obj:
            row["qualidade"] = obj.get("qualidade")
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            f"Loaded 0 usable rows from JSONL dataset: {jsonl_path}. "
            f"Bad lines skipped: {bad_lines}"
        )

    df = df.drop_duplicates(subset=["project", "commit_hash"], keep="first")
    df = df[df["claude_ans"].isin(CLASS_ORDER)].copy()
    return df


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_project_key(project: Any) -> str:
    """Normalize project/repo identifiers so joins don't fail on casing."""
    return _normalize_spaces(str(project or "")).casefold()


def normalize_claude_label(label: Any) -> Optional[str]:
    """Normalize Claude's label to one of the 4 target classes.

    Returns None for labels outside the 4-class set (e.g., 'Sem mensagem', 'Outro').
    """

    if label is None:
        return None

    s = _normalize_spaces(str(label))
    # Remove wrapping quotes/brackets that sometimes sneak in.
    while len(s) >= 2 and (
        (s[0] == s[-1] and s[0] in {'"', "'"})
        or (s[0], s[-1]) in [("(", ")"), ("[", "]"), ("{", "}")]
    ):
        s = _normalize_spaces(s[1:-1])

    # Some files may store the entire tuple string like '("Sim", 82)'.
    m = re.match(r"^\(?\s*(?P<label>[^,]+?)\s*,\s*-?\d+\s*\)?$", s)
    if m:
        return normalize_claude_label(m.group("label"))

    key = _strip_accents(s).casefold()

    # Make parsing robust to punctuation / extra explanation text.
    key = re.sub(r"[\r\n\t]", " ", key)
    key = re.sub(r"[\.,;:!\?\(\)\[\]\{\}\"']", " ", key)
    key = _normalize_spaces(key)

    # Prefer longer matches first.
    if re.search(r"\bprovavelmente\s+sim\b", key):
        return "Provavelmente sim"
    if re.search(r"\bprovavelmente\s+nao\b", key):
        return "Provavelmente não"
    if re.search(r"\bsim\b", key):
        return "Sim"
    if re.search(r"\bnao\b", key):
        return "Não"

    return None


def infer_project_from_filename(path: Path) -> str:
    name = path.stem
    if name.endswith("_rot_gpt"):
        return name[: -len("_rot_gpt")]
    if name.startswith("claude_results_"):
        return name[len("claude_results_") :]
    return name


def load_rot_csvs(csv_dirs: Sequence[Path]) -> pd.DataFrame:
    paths: List[Path] = []
    for d in csv_dirs:
        paths.extend(sorted(d.glob("*_rot_gpt.csv")))
    if not paths:
        raise FileNotFoundError(f"No *_rot_gpt.csv files found under: {list(map(str, csv_dirs))}")

    frames: List[pd.DataFrame] = []
    for p in paths:
        df = pd.read_csv(p, sep=";", encoding="utf-8", engine="python")
        df.columns = [str(c).strip() for c in df.columns]

        # Some files start with an empty first column (Excel-style); drop unnamed index columns.
        drop_unnamed = [c for c in df.columns if not c or str(c).startswith("Unnamed")]
        if drop_unnamed:
            df = df.drop(columns=drop_unnamed)

        df["project"] = normalize_project_key(infer_project_from_filename(p))
        df["commit_hash"] = df["commit_hash"].astype(str).str.strip()
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    return out


def load_claude_labels(claude_dirs: Sequence[Path]) -> pd.DataFrame:
    paths: List[Path] = []
    for d in claude_dirs:
        paths.extend(sorted(d.glob("claude_results_*.json")))
    if not paths:
        raise FileNotFoundError(
            f"No claude_results_*.json files found under: {list(map(str, claude_dirs))}"
        )

    rows: List[Dict[str, Any]] = []

    for p in paths:
        project = normalize_project_key(infer_project_from_filename(p))
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {p}, got {type(data).__name__}")
        for item in data:
            if not isinstance(item, dict):
                continue
            commit_hash = str(item.get("commit_hash") or "").strip()
            if not commit_hash:
                continue
            label = normalize_claude_label(item.get("reflete_mudanca"))
            if not label:
                continue
            rows.append(
                {
                    "project": project,
                    "commit_hash": commit_hash,
                    "claude_ans": label,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(
            "Loaded 0 usable Claude labels after normalization. "
            "Check if labels include only the four required classes."
        )

    # Deduplicate (keep first).
    df = df.drop_duplicates(subset=["project", "commit_hash"], keep="first")
    return df


def load_split_defect_inducing(path: Path, split_name: str) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}, got {type(data).__name__}")

    rows: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        commit_hash = str(item.get("commit_hash") or "").strip()
        repo = str(item.get("repo") or "").strip()
        if not commit_hash or not repo or "/" not in repo:
            continue
        project = normalize_project_key(repo.split("/", 1)[1].strip())  # owner/repo -> repo
        rows.append({"project": project, "commit_hash": commit_hash, "split": split_name})

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["project", "commit_hash"], keep="first")
    return df


def build_labeled_dataset(
    csv_dirs: Sequence[Path],
    claude_dirs: Sequence[Path],
    train_split_path: Path,
    test_split_path: Path,
    *,
    debug_joins: bool = False,
) -> pd.DataFrame:
    feats = load_rot_csvs(csv_dirs)
    labels = load_claude_labels(claude_dirs)
    train_split = load_split_defect_inducing(train_split_path, "train")
    test_split = load_split_defect_inducing(test_split_path, "test")
    splits = pd.concat([train_split, test_split], ignore_index=True)

    if debug_joins:
        print("\n[debug] loaded")
        print(f"  features rows: {len(feats)} projects: {feats['project'].nunique()}")
        print(f"  labels rows:   {len(labels)} projects: {labels['project'].nunique()}")
        print(f"  splits rows:   {len(splits)} projects: {splits['project'].nunique()}")
        print("  features by project (top 20):")
        print(feats["project"].value_counts().head(20).to_string())
        print("  labels by project (top 20):")
        print(labels["project"].value_counts().head(20).to_string())
        print("  splits by project (top 20):")
        print(splits["project"].value_counts().head(20).to_string())

    merged = feats.merge(splits, on=["project", "commit_hash"], how="inner")
    if debug_joins:
        print("\n[debug] after features ⨝ splits")
        print(f"  rows: {len(merged)}")
        print(f"  split counts: {merged['split'].value_counts(dropna=False).to_dict()}")
        print("  by project (top 20):")
        print(merged["project"].value_counts().head(20).to_string())

    merged = merged.merge(labels, on=["project", "commit_hash"], how="inner")
    if debug_joins:
        print("\n[debug] after (features ⨝ splits) ⨝ labels")
        print(f"  rows: {len(merged)}")
        print(f"  split counts: {merged['split'].value_counts(dropna=False).to_dict()}")
        print("  label counts:")
        print(merged["claude_ans"].value_counts(dropna=False).to_string())

    # Drop requested columns if present.
    cols_to_drop = [c for c in DROP_COLUMNS if c in merged.columns]
    if cols_to_drop:
        merged = merged.drop(columns=cols_to_drop)

    # Keep only the expected 4 labels.
    merged = merged[merged["claude_ans"].isin(CLASS_ORDER)].copy()

    if debug_joins:
        print("\n[debug] final dataset")
        print(f"  rows: {len(merged)}")
        print(f"  split counts: {merged['split'].value_counts(dropna=False).to_dict()}")
        print("  label counts:")
        print(merged["claude_ans"].value_counts(dropna=False).to_string())

    return merged


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    # Prefer a combined text field when available.
    text_col = "text" if "text" in X.columns else ("commit_message" if "commit_message" in X.columns else None)

    categorical_cols: List[str] = []
    numeric_cols: List[str] = []

    for col in X.columns:
        if col in {"claude_ans", "split"}:
            continue
        if col == "commit_hash":
            continue
        if col == text_col:
            continue

        series = X[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    transformers: List[Tuple[str, Any, List[str] | str]] = []

    if text_col:
        transformers.append(
            (
                "msg",
                TfidfVectorizer(
                    lowercase=True,
                    max_features=20000,
                    ngram_range=(1, 2),
                ),
                text_col,
            )
        )

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler(with_mean=False)),
                    ]
                ),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns found after filtering.")

    return ColumnTransformer(transformers=transformers)


def evaluate_by_true_label(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> pd.DataFrame:
    recalls = recall_score(y_true, y_pred, labels=list(labels), average=None, zero_division=0)
    recall_by_label = dict(zip(labels, recalls))

    rows: List[Dict[str, Any]] = []
    for lab in labels:
        idx = [i for i, y in enumerate(y_true) if y == lab]
        n = len(idx)
        if n == 0:
            rows.append(
                {
                    "claude_ans": lab,
                    "n_test": 0,
                    "n_errors": 0,
                    "error_rate": 0.0,
                    "recall": 0.0,
                }
            )
            continue
        errors = sum(1 for i in idx if y_pred[i] != y_true[i])
        rows.append(
            {
                "claude_ans": lab,
                "n_test": n,
                "n_errors": errors,
                "error_rate": errors / n,
                "recall": float(recall_by_label.get(lab, 0.0)),
            }
        )

    return pd.DataFrame(rows)


def format_metrics_table(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> pd.DataFrame:
    """Return metrics table in the user's requested layout.

    Columns: claude_ans, total_teste, qtd_erros, qtd_acertos, taxa_erro, acuracia, recall_classe
    """
    recalls = recall_score(
        list(y_true),
        list(y_pred),
        labels=list(labels),
        average=None,
        zero_division=0
    )
    recall_by_label = dict(zip(labels, recalls))

    claude_ans_map = {
        1: "Provavelmente_sim",
        2: "Sim",
        3: "Provavelmente_não",
        4: "Não",
    }

    rows: List[Dict[str, Any]] = []
    for i, lab in enumerate(labels, start=1):
        idxs = [j for j, y in enumerate(y_true) if y == lab]
        n = len(idxs)
        errors = sum(1 for j in idxs if y_pred[j] != y_true[j]) if n > 0 else 0
        correct = n - errors
        taxa_erro = float(errors / n) if n > 0 else 0.0
        acuracia = float(correct / n) if n > 0 else 0.0

        rows.append(
            {
                "claude_ans": claude_ans_map.get(i, str(lab)),
                "total_teste": n,
                "qtd_erros": int(errors),
                "qtd_acertos": int(correct),
                "taxa_erro": taxa_erro,
                "acuracia": acuracia,
                "recall_classe": float(recall_by_label.get(lab, 0.0)),
            }
        )

    return pd.DataFrame(rows)


def _has_usable_split_column(df: pd.DataFrame) -> bool:
    if "split" not in df.columns:
        return False
    splits = set(df["split"].dropna().astype(str).str.strip().tolist())
    return ("train" in splits) and ("test" in splits)


def _derive_train_test_indices(
    df: pd.DataFrame,
    *,
    label_col: str = "claude_ans",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Derive a train/test split from labeled rows.

    This is used as a fallback when the provided split files don't overlap
    with the available Claude labels (common when only part of the dataset
    was labeled).

    Guarantee (when possible): at least 1 example per class in *test* and
    at least 1 example per class in *train*.
    """

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    if df.empty:
        raise ValueError("Cannot split an empty dataset")

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    rng = np.random.default_rng(random_state)
    y = df[label_col].astype(str).tolist()

    by_label: Dict[str, List[int]] = {}
    for i, lab in enumerate(y):
        by_label.setdefault(lab, []).append(i)

    # 1) Seed: 1 test + 1 train per class if possible.
    test_idx: List[int] = []
    train_idx: List[int] = []
    unassigned: List[int] = []

    for lab, idxs in by_label.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        if len(idxs) >= 2:
            test_idx.append(idxs[0])
            train_idx.append(idxs[1])
            unassigned.extend(idxs[2:])
        else:
            # Can't put this class in both splits; keep it in train.
            train_idx.append(idxs[0])

    n = len(df)
    # Ensure at least one item is left for training.
    desired_test = int(round(test_size * n))
    desired_test = max(desired_test, len(test_idx))
    desired_test = min(desired_test, n - 1) if n > 1 else 0

    # 2) Fill remaining test from unassigned pool.
    remaining_need = max(0, desired_test - len(test_idx))
    if remaining_need > 0 and unassigned:
        if remaining_need >= len(unassigned):
            extra = list(unassigned)
            unassigned = []
        else:
            extra = rng.choice(unassigned, size=remaining_need, replace=False).tolist()
            extra_set = set(extra)
            unassigned = [i for i in unassigned if i not in extra_set]
        test_idx.extend(extra)

    # 3) Everything else goes to train.
    train_idx.extend(unassigned)

    # Final sanity: if we still somehow ended with an empty split, fall back to a simple random split.
    if not train_idx or not test_idx:
        all_idx = np.arange(n)
        rng.shuffle(all_idx)
        if n == 1:
            return all_idx, np.array([], dtype=int)
        cut = max(1, min(n - 1, desired_test or 1))
        return all_idx[cut:], all_idx[:cut]

    return np.array(sorted(train_idx), dtype=int), np.array(sorted(test_idx), dtype=int)


def _select_train_test_frames(
    df: pd.DataFrame,
    *,
    fallback_split: str,
    fallback_test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (train_df, test_df, split_source)."""

    if _has_usable_split_column(df):
        train_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()
        if not train_df.empty and not test_df.empty:
            return train_df, test_df, "provided"

    if fallback_split == "none":
        train_df = df[df.get("split") == "train"].copy() if "split" in df.columns else df.iloc[0:0].copy()
        test_df = df[df.get("split") == "test"].copy() if "split" in df.columns else df.copy()
        return train_df, test_df, "none"

    train_idx, test_idx = _derive_train_test_indices(
        df,
        label_col="claude_ans",
        test_size=fallback_test_size,
        random_state=random_state,
    )
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy(), "derived"


def main() -> None:
    root = Path(__file__).resolve().parent

    p = argparse.ArgumentParser(
        description="Prepare dataset + train logistic regression to predict Claude reflection labels."
    )

    p.add_argument(
        "--dataset",
        choices=["rot_csv", "jsonl"],
        default="rot_csv",
        help="Which dataset source to use (default: rot_csv).",
    )
    p.add_argument(
        "--jsonl-path",
        type=Path,
        default=root / "reflection_labeled_commits.jsonl",
        help="Path to reflection_labeled_commits.jsonl (default: REQ1/reflection_labeled_commits.jsonl)",
    )
    p.add_argument(
        "--csv-dir",
        type=Path,
        default=root / "data" / "csvs",
        help="Directory containing *_rot_gpt.csv files (default: REQ1/data/csvs)",
    )
    p.add_argument(
        "--claude-dir",
        type=Path,
        default=root / "data" / "claude results",
        help="Directory containing claude_results_*.json (default: REQ1/data/claude results)",
    )
    p.add_argument(
        "--exclude-old-data",
        action="store_true",
        help="Do not read additional files from REQ1/old data.",
    )
    p.add_argument(
        "--train-split",
        type=Path,
        default=root / "splits" / "train" / "defect_inducing.json",
        help="Train split JSON path (default: REQ1/splits/train/defect_inducing.json)",
    )
    p.add_argument(
        "--test-split",
        type=Path,
        default=root / "splits" / "test" / "defect_inducing.json",
        help="Test split JSON path (default: REQ1/splits/test/defect_inducing.json)",
    )
    p.add_argument(
        "--out-dataset-csv",
        type=Path,
        default=root / "analysis_out" / "reflection_dataset_simplified.csv",
        help="Output simplified labeled dataset CSV",
    )
    p.add_argument(
        "--out-summary-csv",
        type=Path,
        default=root / "analysis_out" / "reflection_dataset_summary.csv",
        help="Output human-readable summary CSV (no full diff/text)",
    )
    p.add_argument(
        "--out-metrics-csv",
        type=Path,
        default=root / "analysis_out" / "reflection_logreg_metrics_by_claude_ans.csv",
        help="Output metrics table CSV",
    )
    p.add_argument(
        "--only-prepare",
        action="store_true",
        help="Only write the simplified labeled CSV; skip training.",
    )

    p.add_argument(
        "--fallback-split",
        choices=["auto", "always", "none"],
        default="auto",
        help=(
            "What to do if the provided split files yield an empty train or test set after merging labels. "
            "auto=derive a split only when needed; always=always derive; none=error out."
        ),
    )
    p.add_argument(
        "--fallback-test-size",
        type=float,
        default=0.2,
        help="Test fraction for derived split (default: 0.2).",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for derived splitting (default: 42).",
    )
    p.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="LogisticRegression class_weight (default: balanced).",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Max iterations for LogisticRegression (default: 5000).",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Tolerance for stopping criteria (default: 1e-4).",
    )
    p.add_argument(
        "--debug-joins",
        action="store_true",
        help="Print per-project row counts before/after each merge step.",
    )

    args = p.parse_args()

    if args.dataset == "jsonl":
        base = load_reflection_jsonl_dataset(args.jsonl_path)
        # Combine commit message + diff into one text column for TF-IDF.
        base["text"] = (base["commit_message"].fillna("") + "\n" + base["diff"].fillna("")).astype(str)
        # Keep split column for compatibility with downstream code.
        y_all = base["claude_ans"].astype(str)

        # If any class has < 2 samples, stratified split is impossible.
        counts = y_all.value_counts()
        can_stratify = (counts.min() >= 2) if not counts.empty else False
        if can_stratify:
            train_idx, test_idx = train_test_split(
                base.index.to_numpy(),
                test_size=args.fallback_test_size,
                random_state=args.random_state,
                stratify=y_all,
            )
            base.loc[train_idx, "split"] = "train"
            base.loc[test_idx, "split"] = "test"
        else:
            # Fallback: keep very rare classes in train; still produce a test set.
            train_idx, test_idx = _derive_train_test_indices(
                base,
                label_col="claude_ans",
                test_size=args.fallback_test_size,
                random_state=args.random_state,
            )
            base["split"] = "train"
            base.loc[base.index[test_idx], "split"] = "test"

        df = base
    else:
        csv_dirs: List[Path] = [args.csv_dir]
        claude_dirs: List[Path] = [args.claude_dir]
        if not args.exclude_old_data:
            old_data = root / "old data"
            if old_data.exists() and old_data.is_dir():
                csv_dirs.append(old_data)
                claude_dirs.append(old_data)

        df = build_labeled_dataset(
            csv_dirs=csv_dirs,
            claude_dirs=claude_dirs,
            train_split_path=args.train_split,
            test_split_path=args.test_split,
            debug_joins=args.debug_joins,
        )

    args.out_dataset_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dataset_csv, index=False)

    # Human-readable summary: keep identifiers, labels, and simple sizes.
    summary_cols = [c for c in ["commit_hash", "project", "claude_ans", "split", "qualidade"] if c in df.columns]
    summary = df[summary_cols].copy() if summary_cols else pd.DataFrame(index=df.index)
    if "commit_message" in df.columns:
        summary["commit_message"] = df["commit_message"].fillna("").astype(str)
        summary["commit_message_len"] = summary["commit_message"].str.len()
    if "diff" in df.columns:
        diff_s = df["diff"].fillna("").astype(str)
        summary["diff_len"] = diff_s.str.len()
        summary["diff_lines"] = diff_s.str.count("\n") + diff_s.ne("").astype(int)

    args.out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary_csv, index=False)

    print("Prepared dataset")
    print(f"  rows: {len(df)}")
    print(f"  splits: {df['split'].value_counts(dropna=False).to_dict()}")
    print(f"  labels: {df['claude_ans'].value_counts(dropna=False).to_dict()}")
    print(f"  wrote: {args.out_dataset_csv}")
    print(f"  wrote summary: {args.out_summary_csv}")

    if args.only_prepare:
        return

    want_fallback = args.fallback_split
    if want_fallback == "auto":
        want_fallback = "always" if (not _has_usable_split_column(df)) else "none"
        # If split exists but yields empties, we'll detect below and derive.
        if _has_usable_split_column(df):
            if (df["split"] == "train").sum() == 0 or (df["split"] == "test").sum() == 0:
                want_fallback = "always"

    train_df, test_df, split_source = _select_train_test_frames(
        df,
        fallback_split=want_fallback,
        fallback_test_size=args.fallback_test_size,
        random_state=args.random_state,
    )

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Need non-empty train and test sets to train/evaluate. "
            f"train={len(train_df)} test={len(test_df)}. "
            "If you only have labels for a single split, rerun with --fallback-split always."
        )

    if split_source == "derived":
        print(
            "\nNOTE: Provided split files did not yield both train and test after merging labels. "
            "Training/evaluation will use a derived split from the labeled rows." 
        )
        print(
            f"  derived splits: train={len(train_df)} test={len(test_df)} "
            f"(test_size={args.fallback_test_size}, random_state={args.random_state})"
        )

    y_train = train_df["claude_ans"].tolist()
    y_test = test_df["claude_ans"].tolist()

    X_train = train_df.drop(columns=["claude_ans"])
    X_test = test_df.drop(columns=["claude_ans"])

    pre = make_preprocessor(X_train)
    class_weight = None if args.class_weight == "none" else "balanced"
    clf = LogisticRegression(
        solver="saga",
        max_iter=args.max_iter,
        n_jobs=-1,
        class_weight=class_weight,
        tol=args.tol,
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    table = format_metrics_table(y_test, list(y_pred), CLASS_ORDER)
    args.out_metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_metrics_csv, index=False)

    print("\nMetrics by true claude_ans (test)")
    print(table.to_string(index=False))
    print(f"\nWrote metrics -> {args.out_metrics_csv}")

    print("\nClassification report")
    print(classification_report(y_test, y_pred, labels=list(CLASS_ORDER), zero_division=0))

    print("Confusion matrix (rows=true, cols=pred)")
    cm = confusion_matrix(y_test, y_pred, labels=list(CLASS_ORDER))
    cm_df = pd.DataFrame(cm, index=[f"true:{c}" for c in CLASS_ORDER], columns=[f"pred:{c}" for c in CLASS_ORDER])
    print(cm_df.to_string())


if __name__ == "__main__":
    main()
