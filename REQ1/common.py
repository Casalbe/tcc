"""
common.py
=========
Funções compartilhadas entre os classificadores (Random Forest e
Logistic Regression): carregamento de dados, split estratificado,
matriz de features e métricas auxiliares.

Mantido separado para que os dois scripts de treino usem EXATAMENTE
a mesma lógica de dados — condição necessária para uma comparação
estatística justa entre os dois modelos.
"""

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

# ─── Configuration ────────────────────────────────────────────────────────────

TRAIN_DIR = Path("splits/test")

NUMERIC_FEATURES = [
    "fix", "ns", "nd", "nf", "entropy",
    "la", "ld", "lt", "ndev", "age",
    "nuc", "exp", "rexp", "sexp",
]

REFLETE_ORDER = ["Sim", "Provavelmente sim", "Provavelmente não", "Não"]
REFLETE_SCORE = {
    "Sim": 3,
    "Provavelmente sim": 2,
    "Provavelmente não": 1,
    "Não": 0,
}

TEST_SIZE = 0.30

# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_json_file(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def safe_float(value, default=0.0) -> float:
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


def gmean_recall(y_true, y_pred) -> float:
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    return float(np.sqrt(np.prod(recalls)))


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_reflete_mudanca(value) -> str | None:
    if value is None:
        return None

    text = " ".join(str(value).strip().split())
    if not text:
        return None

    key = _strip_accents(text).lower()
    mapping = {
        "sim": "Sim",
        "provavelmente sim": "Provavelmente sim",
        "provavelmente nao": "Provavelmente não",
        "nao": "Não",
        "sem mensagem": "Sem mensagem",
    }
    return mapping.get(key, text)


def reflete_to_score(value) -> int | None:
    label = normalize_reflete_mudanca(value)
    if label is None:
        return None
    return REFLETE_SCORE.get(label)

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_all_data(directory: Path = TRAIN_DIR) -> pd.DataFrame:
    """
    Carrega complete_clean.json + complete_defect_inducing.json e retorna
    um DataFrame NÃO embaralhado (a ordem é determinística: clean primeiro,
    depois defect). O embaralhamento/seed fica a cargo de quem chama esta
    função (cada run do experimento decide seu próprio random_state),
    para que o carregamento em si seja 100% reprodutível.
    """
    clean_path  = directory / "complete_clean.json"
    defect_path = directory / "complete_defect_inducing.json"

    for p in (clean_path, defect_path):
        if not p.exists():
            sys.exit(f"[ERROR] Required file not found: {p}")

    clean_records  = load_json_file(clean_path)
    defect_records = load_json_file(defect_path)

    print(f"  complete_clean:           {len(clean_records):,} records")
    print(f"  complete_defect_inducing: {len(defect_records):,} records")

    rows = []
    for r in clean_records + defect_records:
        row = {
            "label":           contains_bug_label(r),
            "commit_hash":     r.get("commit_hash", ""),
            "repo":            r.get("repo", ""),
            "reflete_mudanca": r.get("reflete_mudanca"),
        }
        for feat in NUMERIC_FEATURES:
            row[feat] = safe_float(r.get(feat))
        rows.append(row)

    return pd.DataFrame(rows)


def shuffle_data(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Embaralha o DataFrame de forma determinística dado um seed."""
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def stratified_split(df: pd.DataFrame, seed: int, test_size: float = TEST_SIZE):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# ─── Feature Matrix ───────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[NUMERIC_FEATURES].values.astype(np.float32)


def build_prediction_log(df: pd.DataFrame, model: str, seed: int, y_proba) -> pd.DataFrame:
    rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        rows.append({
            "model": model,
            "seed": seed,
            "commit_hash": row.get("commit_hash", ""),
            "repo": row.get("repo", ""),
            "label": int(row["label"]),
            "pred": int(row["pred"]),
            "pred_bug_proba": float(y_proba[idx]) if y_proba is not None else float("nan"),
            "correct": int(int(row["label"]) == int(row["pred"])),
            "reflete_mudanca": row.get("reflete_mudanca"),
            "reflete_score": reflete_to_score(row.get("reflete_mudanca")),
        })
    return pd.DataFrame(rows)


def append_prediction_log(df: pd.DataFrame, model: str, seed: int, y_proba, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_df = build_prediction_log(df, model, seed, y_proba)
    write_header = not out_path.exists()
    log_df.to_csv(out_path, mode="a", header=write_header, index=False)
