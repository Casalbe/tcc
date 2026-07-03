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
