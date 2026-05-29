import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None


LABELS_ORDERED = ["Não", "Provavelmente não", "Provavelmente sim", "Sim"]


_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_IDENT_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]{1,}\b")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_label(label: str) -> Optional[str]:
    label = (label or "").strip()
    if not label:
        return None
    # tolerate case / accents issues in upstream outputs
    label_norm = label.lower()
    mapping = {
        "nao": "Não",
        "não": "Não",
        "provavelmente nao": "Provavelmente não",
        "provavelmente não": "Provavelmente não",
        "provavelmente sim": "Provavelmente sim",
        "sim": "Sim",
    }
    # remove duplicate whitespace
    label_norm = " ".join(label_norm.split())
    return mapping.get(label_norm)


def truncate_text(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def tokenize_identifiers(text: str) -> List[str]:
    return _IDENT_RE.findall(text or "")


def diff_stats(diff_text: str) -> Tuple[int, int, int, int]:
    added = 0
    removed = 0
    hunks = 0
    files = 0
    for line in (diff_text or "").splitlines():
        if line.startswith("diff --git "):
            files += 1
        if line.startswith("@@"):
            hunks += 1
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return files, hunks, added, removed


def overlap_features(message: str, diff_text: str) -> Tuple[float, float]:
    msg_words = set(tokenize_words(message))
    diff_words = set(tokenize_words(diff_text))
    if not msg_words or not diff_words:
        return 0.0, 0.0
    inter = msg_words & diff_words
    union = msg_words | diff_words
    jaccard = len(inter) / max(1, len(union))
    recall = len(inter) / max(1, len(msg_words))
    return float(jaccard), float(recall)


@dataclass
class FeatureBundle:
    vec_msg: TfidfVectorizer
    vec_diff: TfidfVectorizer
    label_to_id: Dict[str, int]


def build_features(
    rows: List[Dict[str, Any]],
    vec_msg: Optional[TfidfVectorizer] = None,
    vec_diff: Optional[TfidfVectorizer] = None,
    fit: bool = False,
    max_diff_chars: int = 6000,
) -> Tuple[sparse.csr_matrix, np.ndarray, FeatureBundle]:

    # Drop rows with out-of-scope labels (e.g., upstream "Sem mensagem")
    kept_rows: List[Dict[str, Any]] = []
    labels: List[str] = []
    dropped = 0
    for r in rows:
        lab = normalize_label(str(r.get("reflete_mudanca") or ""))
        if lab is None:
            dropped += 1
            continue
        kept_rows.append(r)
        labels.append(lab)

    if dropped:
        print(f"[train_reflection_models] Dropped {dropped} rows with unknown labels")

    rows = kept_rows
    messages = [str(r.get("commit_message") or "") for r in rows]
    diffs = [truncate_text(str(r.get("diff") or ""), max_diff_chars) for r in rows]

    label_to_id = {lab: i for i, lab in enumerate(LABELS_ORDERED)}
    y = np.array([label_to_id[l] for l in labels], dtype=np.int32)

    if vec_msg is None:
        vec_msg = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=60000,
        )
    if vec_diff is None:
        vec_diff = TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b[\w\-/\.]{2,}\b",
            ngram_range=(1, 1),
            min_df=2,
            max_features=120000,
        )

    if fit:
        X_msg = vec_msg.fit_transform(messages)
        X_diff = vec_diff.fit_transform(diffs)
    else:
        X_msg = vec_msg.transform(messages)
        X_diff = vec_diff.transform(diffs)

    numeric = []
    for m, d in zip(messages, diffs):
        files, hunks, added, removed = diff_stats(d)
        jac, recall = overlap_features(m, d)
        numeric.append(
            [
                len(m),
                len(tokenize_words(m)),
                len(d),
                files,
                hunks,
                added,
                removed,
                jac,
                recall,
            ]
        )

    X_num = sparse.csr_matrix(np.array(numeric, dtype=np.float32))
    X = sparse.hstack([X_msg, X_diff, X_num], format="csr")
    bundle = FeatureBundle(vec_msg=vec_msg, vec_diff=vec_diff, label_to_id=label_to_id)
    return X, y, bundle


def eval_and_save_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    prefix: str,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=LABELS_ORDERED, output_dict=True)

    # Save confusion matrix as CSV
    cm_path = out_dir / f"{prefix}_confusion_matrix.csv"
    with cm_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("," + ",".join(LABELS_ORDERED) + "\n")
        for i, row in enumerate(cm.tolist()):
            f.write(LABELS_ORDERED[i] + "," + ",".join(str(x) for x in row) + "\n")

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "classification_report": report,
    }
    (out_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train commit-message reflection classifiers (LR, Linear SVM, XGBoost) from the labeled JSONL dataset."
    )
    p.add_argument(
        "--data",
        default=str(Path("REQ1") / "data" / "reflection_labeled_commits.jsonl"),
        help="Path to labeled JSONL produced by build_reflection_labeled_dataset.py",
    )
    p.add_argument(
        "--out-dir",
        default=str(Path("REQ1") / "models" / "reflection"),
        help="Directory to save models and vectorizers",
    )
    p.add_argument(
        "--metrics-dir",
        default=str(Path("REQ1") / "analysis_out" / "reflection_models"),
        help="Directory to save evaluation artifacts",
    )
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-diff-chars", type=int, default=6000)

    args = p.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    metrics_dir = Path(args.metrics_dir)

    rows = _read_jsonl(data_path)
    if not rows:
        raise SystemExit(f"Empty dataset: {data_path}")

    X, y, bundle = build_features(rows, fit=True, max_diff_chars=args.max_diff_chars)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Save feature bundle (vectorizers + label mapping)
    joblib.dump(bundle, out_dir / "feature_bundle.joblib")

    results: Dict[str, Any] = {}

    # Logistic Regression
    lr = LogisticRegression(
        max_iter=3000,
        n_jobs=1,
        solver="lbfgs",
        class_weight="balanced",
    )
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    results["logreg"] = eval_and_save_metrics(y_test, pred_lr, metrics_dir, "logreg")
    joblib.dump(lr, out_dir / "logreg.joblib")

    # Linear SVM
    svm = LinearSVC(class_weight="balanced")
    svm.fit(X_train, y_train)
    pred_svm = svm.predict(X_test)
    results["linear_svm"] = eval_and_save_metrics(y_test, pred_svm, metrics_dir, "linear_svm")
    joblib.dump(svm, out_dir / "linear_svm.joblib")

    # XGBoost (optional dependency)
    if xgb is not None:
        xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(LABELS_ORDERED),
            n_estimators=400,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.7,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="mlogloss",
            random_state=args.seed,
        )
        xgb_model.fit(X_train, y_train)
        pred_xgb = xgb_model.predict(X_test)
        results["xgboost"] = eval_and_save_metrics(y_test, pred_xgb, metrics_dir, "xgboost")
        joblib.dump(xgb_model, out_dir / "xgboost.joblib")
    else:
        results["xgboost"] = {
            "skipped": True,
            "reason": "xgboost is not installed in this environment",
        }

    (metrics_dir / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Done. Models saved to: {out_dir}")
    print(f"Metrics saved to: {metrics_dir}")


if __name__ == "__main__":
    main()
