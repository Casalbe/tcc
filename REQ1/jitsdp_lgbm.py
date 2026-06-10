"""
Bug-Inducing Commit Classifier — LightGBM (numeric features only)
==================================================================
Predicts whether a commit introduces a bug based on software-engineering
metrics extracted from the commit history.

Data layout expected:
  splits/train/
    complete_clean.json             (contains_bug: false)
    complete_defect_inducing.json   (contains_bug: true)
  splits/test/
    complete_clean.json             (contains_bug: false, includes reflete_mudanca)
    complete_defect_inducing.json   (contains_bug: true,  includes reflete_mudanca)

The reflete_mudanca field is read directly from the test JSON records —
run merge_reflete.py once before training to stitch it in.

Numeric features used (14 total):
  fix, ns, nd, nf, entropy, la, ld, lt, ndev, age, nuc, exp, rexp, sexp

Outputs
-------
- Trained model saved to  models/lgbm_bug_classifier.joblib
- Console tables:
    1. Overall classification report
    2. Per-stratum table  (reflete_mudanca × contains_bug recall / errors)
    3. Feature importances

Dependencies
------------
pip install lightgbm scikit-learn pandas joblib
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
)

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

TRAIN_DIR = Path("splits/train")
TEST_DIR  = Path("splits/test")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "fix", "ns", "nd", "nf", "entropy",
    "la", "ld", "lt", "ndev", "age",
    "nuc", "exp", "rexp", "sexp",
]

REFLETE_ORDER = ["Sim", "Provavelmente sim", "Provavelmente não", "Não"]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_json_file(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def safe_float(value, default=0.0) -> float:
    """Convert a value to float, returning default on None/empty/error."""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def contains_bug_label(record: dict) -> int:
    val = record.get("contains_bug", False)
    if isinstance(val, bool):
        return int(val)
    return int(str(val).lower() in ("true", "1", "yes"))

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_split(directory: Path, split_name: str, include_reflete: bool = False) -> pd.DataFrame:
    """
    Load complete_clean + complete_defect_inducing JSON files for a split.
    Pass include_reflete=True for the test split to carry the reflete_mudanca
    field through to the DataFrame for stratified evaluation.
    """
    clean_path  = directory / "complete_clean.json"
    defect_path = directory / "complete_defect_inducing.json"

    for p in (clean_path, defect_path):
        if not p.exists():
            sys.exit(f"[ERROR] Required file not found: {p}")

    clean_records  = load_json_file(clean_path)
    defect_records = load_json_file(defect_path)

    print(f"  {split_name}/complete_clean:           {len(clean_records):,} records")
    print(f"  {split_name}/complete_defect_inducing: {len(defect_records):,} records")

    if include_reflete:
        # Count how many records actually carry the annotation
        annotated = sum(
            1 for r in clean_records + defect_records
            if r.get("reflete_mudanca") is not None
        )
        print(f"  {split_name}: {annotated:,} records have reflete_mudanca annotation")
        if annotated == 0:
            print("  [WARNING] No reflete_mudanca values found in the test JSONs.")
            print("            Run merge_reflete.py first to stitch in the annotations.")

    rows = []
    for r in clean_records + defect_records:
        row = {
            "label":       contains_bug_label(r),
            "commit_hash": r.get("commit_hash", ""),
            "repo":        r.get("repo", ""),
        }
        if include_reflete:
            row["reflete_mudanca"] = r.get("reflete_mudanca")  # None if absent
        for feat in NUMERIC_FEATURES:
            row[feat] = safe_float(r.get(feat))
        rows.append(row)

    return pd.DataFrame(rows)

# ─── Training ─────────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return the numeric feature matrix as a dense numpy array."""
    return df[NUMERIC_FEATURES].values.astype(np.float32)


def train(train_df: pd.DataFrame) -> LGBMClassifier:
    """Fit a LGBMClassifier on the numeric feature matrix."""
    X_train = build_feature_matrix(train_df)
    y_train = train_df["label"].values

    print(f"[TRAIN] Feature matrix: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    print(f"[TRAIN] Features: {NUMERIC_FEATURES}")
    print("[TRAIN] Fitting LightGBM …")

    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbosity=-1,
    )
    clf.fit(X_train, y_train)
    print("[TRAIN] Done.\n")
    return clf

# ─── Evaluation ───────────────────────────────────────────────────────────────

def predict(clf: LGBMClassifier, df: pd.DataFrame) -> np.ndarray:
    return clf.predict(build_feature_matrix(df))


def overall_report(y_true, y_pred):
    print("=" * 60)
    print("OVERALL TEST SET CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        y_true, y_pred,
        target_names=["clean (0)", "bug-inducing (1)"],
        digits=4,
    ))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm_df = pd.DataFrame(
        cm,
        index=["Actual clean (0)", "Actual bug (1)"],
        columns=["Pred clean (0)", "Pred bug (1)"],
    )
    print(cm_df.to_string())
    print()


def feature_importance_report(clf: LGBMClassifier):
    print("=" * 60)
    print("FEATURE IMPORTANCES (gain)")
    print("=" * 60)
    feat_df = pd.DataFrame({
        "feature":    NUMERIC_FEATURES,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(feat_df.to_string(index=False))
    print()


def stratified_report(test_df: pd.DataFrame):
    """
    Build a per-stratum evaluation table with one row per reflete_mudanca value.
    Reads reflete_mudanca directly from the test_df column (already loaded from
    the JSON records — no separate annotation dict needed).

    Columns:
      reflete_mudanca — annotation label
      total_teste     — total samples in the stratum
      qtd_erros       — misclassified samples
      qtd_acertos     — correctly classified samples
      taxa_erro       — error rate  (qtd_erros / total_teste)
      acuracia        — accuracy    (qtd_acertos / total_teste)
      recall_classe_0 — recall for the clean class (0)
      recall_classe_1 — recall for the bug-inducing class (1)
    """
    annotated = test_df[test_df["reflete_mudanca"].notna()].copy()
    missing   = test_df[test_df["reflete_mudanca"].isna()]

    print(f"[INFO] Commits with reflete_mudanca annotation : {len(annotated):,}")
    print(f"[INFO] Commits without annotation (excluded)   : {len(missing):,}")
    print()

    if annotated.empty:
        print("[WARNING] No annotated commits found — skipping stratified analysis.")
        return

    rows = []
    for reflete_val in REFLETE_ORDER:
        stratum = annotated[annotated["reflete_mudanca"] == reflete_val]
        if stratum.empty:
            continue

        total   = len(stratum)
        errors  = (stratum["pred"] != stratum["label"]).sum()
        correct = total - errors

        def stratum_recall(cls: int) -> float:
            if not (stratum["label"] == cls).any():
                return float("nan")
            return recall_score(
                stratum["label"].values,
                stratum["pred"].values,
                labels=[cls],
                average="macro",
                zero_division=0,
            )

        rows.append({
            "reflete_mudanca": reflete_val,
            "total_teste":     total,
            "qtd_erros":       int(errors),
            "qtd_acertos":     int(correct),
            "taxa_erro":       round(errors / total, 4),
            "acuracia":        round(correct / total, 4),
            "recall_classe_0": round(stratum_recall(0), 4),
            "recall_classe_1": round(stratum_recall(1), 4),
        })

    results_df = pd.DataFrame(rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("=" * 100)
    print("STRATIFIED RESULTS BY reflete_mudanca")
    print("=" * 100)
    print(results_df.to_string(index=False))
    print()

    out_path = MODEL_DIR / "stratified_results_lgbm.csv"
    results_df.to_csv(out_path, index=False)
    print(f"[INFO] Stratified results saved → {out_path}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BUG-INDUCING COMMIT CLASSIFIER — LIGHTGBM")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[LOAD] Training data …")
    train_df = load_split(TRAIN_DIR, "train", include_reflete=False)

    print("\n[LOAD] Test data …")
    test_df = load_split(TEST_DIR, "test", include_reflete=True)

    # ── Train ──────────────────────────────────────────────────────────────
    clf = train(train_df)

    # ── Persist ────────────────────────────────────────────────────────────
    dump(clf, MODEL_DIR / "lgbm_bug_classifier.joblib")
    print(f"[SAVE] Model saved to {MODEL_DIR}/lgbm_bug_classifier.joblib\n")

    # ── Evaluate ───────────────────────────────────────────────────────────
    test_df["pred"] = predict(clf, test_df)
    overall_report(test_df["label"].values, test_df["pred"].values)
    feature_importance_report(clf)
    stratified_report(test_df)

    print("Done.")


if __name__ == "__main__":
    main()