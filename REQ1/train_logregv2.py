"""
Logistic Regression classifier for commit message quality assessment.

Task: Given a commit message + diff, classify whether the message
      accurately reflects the changes (4 classes):
      - Sim
      - Provavelmente sim
      - Não
      - Provavelmente não

Dataset split:
  Train: 1000 clean commits + 1000 sampled bug-inducing commits (balanced, 2000 total)
  Test:  Remaining 1500 bug-inducing commits
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import joblib


# ── Constants ────────────────────────────────────────────────────────────────

LABEL_COLUMN = "reflete_mudanca"
CLASSES = ["Sim", "Provavelmente sim", "Provavelmente não", "Não"]

# How to combine commit_message and diff into a single text feature.
# Options: "message_only" | "diff_only" | "concat"
TEXT_MODE = "concat"

RANDOM_SEED = 42


# ── I/O helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def build_text_series(df: pd.DataFrame, mode: str = TEXT_MODE) -> pd.Series:
    """Combine commit_message and diff into a single text Series."""
    if mode == "message_only":
        return df["commit_message"].fillna("")
    if mode == "diff_only":
        return df["diff"].fillna("")
    # concat (default): separator helps the vectoriser distinguish sections
    return "MESSAGE: " + df["commit_message"].fillna("") + "\nDIFF: " + df["diff"].fillna("")


# ── Dataset construction ──────────────────────────────────────────────────────

def build_datasets(
    clean_path: str,
    defect_path: str,
    n_defect_train: int = 1000,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Returns (X_train, y_train, X_test, y_test) as pandas Series.

    Train: all clean + n_defect_train sampled bug-inducing commits.
    Test:  remaining bug-inducing commits.
    """
    df_clean  = load_jsonl(clean_path)
    df_defect = load_jsonl(defect_path)

    print(f"Loaded {len(df_clean):,} clean commits.")
    print(f"Loaded {len(df_defect):,} defect-inducing commits.")

    # Shuffle defect and split train / test
    df_defect = df_defect.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_defect_train = df_defect.iloc[:n_defect_train]
    df_defect_test  = df_defect.iloc[n_defect_train:]

    df_train = (
        pd.concat([df_clean, df_defect_train], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    print(f"\nTrain size : {len(df_train):,}  "
          f"(clean={len(df_clean)}, defect={len(df_defect_train)})")
    print(f"Test size  : {len(df_defect_test):,}  (all defect-inducing)\n")

    X_train = build_text_series(df_train)
    y_train = df_train[LABEL_COLUMN]

    X_test = build_text_series(df_defect_test)
    y_test = df_defect_test[LABEL_COLUMN]

    return X_train, y_train, X_test, y_test


# ── Model ────────────────────────────────────────────────────────────────────

def build_pipeline(
    max_features: int = 50_000,
    ngram_range: tuple = (1, 2),
    C: float = 1.0,
    max_iter: int = 1000,
    solver: str = "lbfgs",
    class_weight: str | None = "balanced",
) -> Pipeline:
    """
    TF-IDF (char + word n-grams) → Logistic Regression.

    Two vectorisers are used:
      - word n-grams  (1-2): captures token-level patterns
      - char n-grams  (3-5): robust to typos / code tokens / camelCase
    They are concatenated via FeatureUnion.
    """
    from sklearn.pipeline import FeatureUnion

    tfidf_word = TfidfVectorizer(
        analyzer="word",
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
    )
    tfidf_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=max_features // 2,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    features = FeatureUnion([
        ("word", tfidf_word),
        ("char", tfidf_char),
    ])

    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        class_weight=class_weight,
        multi_class="multinomial",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    return Pipeline([("features", features), ("clf", clf)])


# ── Evaluation helpers ────────────────────────────────────────────────────────

def build_per_class_table(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Build a per-class results DataFrame with columns:
      answer | total_teste | qtd_erros | qtd_acertos | taxa_erro | acuracia | recall
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    rows = []
    for cls in CLASSES:
        mask        = y_true == cls
        total       = mask.sum()
        acertos     = ((y_true == y_pred) & mask).sum()
        erros       = total - acertos
        taxa_erro   = erros  / total if total else 0.0
        acuracia    = acertos / total if total else 0.0
        # recall: TP / (TP + FN) — among all true positives of this class,
        # how many did the model label correctly?
        recall      = acertos / total if total else 0.0   # same as per-class accuracy
        rows.append({
            "answer"      : cls,
            "total_teste" : int(total),
            "qtd_erros"   : int(erros),
            "qtd_acertos" : int(acertos),
            "taxa_erro"   : round(taxa_erro, 4),
            "acuracia"    : round(acuracia,  4),
            "recall"      : round(recall,    4),
        })

    # Append a totals row
    total_all   = len(y_true)
    acertos_all = (y_true == y_pred).sum()
    erros_all   = total_all - acertos_all
    rows.append({
        "answer"      : "TOTAL",
        "total_teste" : int(total_all),
        "qtd_erros"   : int(erros_all),
        "qtd_acertos" : int(acertos_all),
        "taxa_erro"   : round(erros_all  / total_all, 4),
        "acuracia"    : round(acertos_all / total_all, 4),
        "recall"      : round(recall_score(y_true, y_pred, labels=CLASSES,
                                           average="macro", zero_division=0), 4),
    })

    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame, title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print()


def save_confusion_matrix(y_true, y_pred, out_path: str, split: str = "Test"):
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=30)
    ax.set_title(f"Confusion Matrix – {split} set")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Logistic Regression classifier for commit message quality."
    )
    parser.add_argument(
        "--clean", required=True,
        help="Path to JSONL with clean (non-bug-inducing) commits (≈1000 items)."
    )
    parser.add_argument(
        "--defect", required=True,
        help="Path to JSONL with bug-inducing commits (≈2500 items)."
    )
    parser.add_argument(
        "--n-defect-train", type=int, default=1000,
        help="Number of defect commits to include in training (default: 1000)."
    )
    parser.add_argument(
        "--text-mode", choices=["message_only", "diff_only", "concat"],
        default=TEXT_MODE,
        help="How to build the text feature from each record (default: concat)."
    )
    parser.add_argument(
        "--C", type=float, default=1.0,
        help="Logistic Regression regularisation strength (default: 1.0)."
    )
    parser.add_argument(
        "--max-features", type=int, default=50_000,
        help="Max TF-IDF features for word n-grams (default: 50 000)."
    )
    parser.add_argument(
        "--ngram-max", type=int, default=2,
        help="Upper bound of word n-gram range (default: 2)."
    )
    parser.add_argument(
        "--max-iter", type=int, default=1000,
        help="Max solver iterations (default: 1000)."
    )
    parser.add_argument(
        "--no-class-weight", action="store_true",
        help="Disable balanced class weighting."
    )
    parser.add_argument(
        "--out-dir", default=".",
        help="Directory to save the model and plots (default: current dir)."
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed (default: 42)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global TEXT_MODE
    TEXT_MODE = args.text_mode

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    X_train, y_train, X_test, y_test = build_datasets(
        clean_path=args.clean,
        defect_path=args.defect,
        n_defect_train=args.n_defect_train,
        seed=args.seed,
    )

    print("Label distribution – train:")
    print(pd.Series(y_train.values).value_counts().reindex(CLASSES).to_string())

    print("\nLabel distribution – test:")
    print(pd.Series(y_test.values).value_counts().reindex(CLASSES).to_string())

    # ── Train ─────────────────────────────────────────────────────────────────
    class_weight = None if args.no_class_weight else "balanced"
    pipeline = build_pipeline(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        C=args.C,
        max_iter=args.max_iter,
        class_weight=class_weight,
    )

    print("\nTraining …")
    pipeline.fit(X_train, y_train)
    print("Done.")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_train_pred = pipeline.predict(X_train)
    train_table  = build_per_class_table(y_train, y_train_pred)
    print_table(train_table, "Train results – per-class breakdown")

    train_csv = out_dir / "results_train.csv"
    train_table.to_csv(train_csv, index=False)
    print(f"  Train table saved → {train_csv}")

    save_confusion_matrix(
        y_train, y_train_pred,
        out_path=str(out_dir / "cm_train.png"),
        split="Train",
    )

    y_test_pred = pipeline.predict(X_test)
    test_table  = build_per_class_table(y_test, y_test_pred)
    print_table(test_table, "Test results – per-class breakdown")

    test_csv = out_dir / "results_test.csv"
    test_table.to_csv(test_csv, index=False)
    print(f"  Test table saved  → {test_csv}")

    save_confusion_matrix(
        y_test, y_test_pred,
        out_path=str(out_dir / "cm_test.png"),
        split="Test",
    )

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = out_dir / "logreg_commit_classifier.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved → {model_path}")

    # ── Top features per class ────────────────────────────────────────────────
    print("\nTop 15 features per class:")
    feature_names = pipeline["features"].get_feature_names_out()
    coef = pipeline["clf"].coef_           # shape: (n_classes, n_features)
    classes = pipeline["clf"].classes_

    for i, cls in enumerate(classes):
        top_idx = np.argsort(coef[i])[-15:][::-1]
        top_feats = [(feature_names[j], round(coef[i][j], 4)) for j in top_idx]
        print(f"\n  [{cls}]")
        for feat, weight in top_feats:
            print(f"    {feat:<35} {weight:+.4f}")


if __name__ == "__main__":
    main()