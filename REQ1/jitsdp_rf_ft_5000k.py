"""
Bug-Inducing Commit Classifier — Random Forest + 10-Fold CV Tuning
==================================================================
- Split estratificado 70/30 a partir do diretório de treino
- Progresso em tempo real candidato a candidato (ParameterGrid loop)
- Tabela estratificada por reflete_mudanca: acerto/erro na classe bug-inducing
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    make_scorer,
    recall_score,
)

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

TRAIN_DIR    = Path("splits/test")
MODEL_DIR    = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TEST_SIZE    = 0.30
RANDOM_STATE = 42
N_FOLDS      = 10

NUMERIC_FEATURES = [
    "fix", "ns", "nd", "nf", "entropy",
    "la", "ld", "lt", "ndev", "age",
    "nuc", "exp", "rexp", "sexp",
]

REFLETE_ORDER = ["Sim", "Provavelmente sim", "Provavelmente não", "Não"]

PARAM_GRID = {
    "n_estimators":      [100, 200],
    "max_depth":         [None, 3, 10, 21],
    "min_samples_split": [2, 10],
    "min_samples_leaf":  [1, 4],
    "max_features":      ["sqrt", "log2"],
}

def gmean_recall(y_true, y_pred) -> float:
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    return float(np.sqrt(np.prod(recalls)))

TUNING_SCORE_NAME = "gmean_recall"
TUNING_SCORE      = make_scorer(gmean_recall)

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

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_all_data(directory: Path) -> pd.DataFrame:
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
    
    from random import randint
    
    df =  pd.DataFrame(rows)
    df = df.sample(frac=1, random_state= randint(1,10000)).reset_index(drop=True)    

    return df 


def stratified_split(df: pd.DataFrame, test_size: float, random_state: int):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

# ─── Feature Matrix ───────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[NUMERIC_FEATURES].values.astype(np.float32)

# ─── Training com progresso em tempo real ─────────────────────────────────────

def train_with_cv(train_df: pd.DataFrame):
    """
    Itera manualmente sobre ParameterGrid para exibir o resultado de cada
    candidato assim que seus N_FOLDS folds terminam.

    Cada RandomForestClassifier usa n_jobs=-1 internamente, então as árvores
    de cada fold ainda rodam em paralelo — apenas a ordem entre candidatos
    é sequencial, o que é necessário para o progresso em tempo real.
    """
    X_train = build_feature_matrix(train_df)
    y_train = train_df["label"].values

    print(f"[TRAIN] Feature matrix : {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    print(f"[TRAIN] Features       : {NUMERIC_FEATURES}")
    print(f"[TRAIN] Distribuição   : "
          f"clean={int((y_train == 0).sum()):,}, bug={int((y_train == 1).sum()):,}\n")

    cv         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    candidates = list(ParameterGrid(PARAM_GRID))
    total      = len(candidates)
    total_fits = total * N_FOLDS

    print(f"[TUNE] {total} candidatos × {N_FOLDS} folds = {total_fits} fits totais")
    print(f"[TUNE] Scoring: {TUNING_SCORE_NAME}")
    print(f"[TUNE] n_jobs=-1 dentro de cada RandomForest\n")

    # Cabeçalho da tabela de progresso
    header = (
        f"  {'#':>3}  {'progresso':>12}  {'gmean_recall':>13}  "
        f"{'±std':>7}  {'ETA':>7}  params"
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    best_score  = -np.inf
    best_params = None
    best_clf    = None
    results     = []
    start       = time.perf_counter()

    for idx, params in enumerate(candidates, start=1):
        t0 = time.perf_counter()

        clf = RandomForestClassifier(
            **params,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

        scores = cross_val_score(
            clf, X_train, y_train,
            cv=cv,
            scoring=TUNING_SCORE,
            n_jobs=1,   # folds sequenciais → garante progresso linha a linha
        )

        mean_score = float(scores.mean())
        std_score  = float(scores.std())
        elapsed    = time.perf_counter() - start
        eta        = (elapsed / idx) * (total - idx)

        is_best = mean_score > best_score
        if is_best:
            best_score  = mean_score
            best_params = params
            # Treina o melhor modelo acumulado no conjunto completo
            best_clf = RandomForestClassifier(
                **params,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ).fit(X_train, y_train)

        marker     = " ◀ BEST" if is_best else ""
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        pct        = 100.0 * idx / total

        print(
            f"  [{idx:>3}/{total}]"
            f"  fits={idx * N_FOLDS:>4}/{total_fits} ({pct:5.1f}%)"
            f"  {mean_score:.4f} ±{std_score:.4f}"
            f"  ETA={eta:5.0f}s"
            f"  | {params_str}"
            f"{marker}"
        )

        results.append({
            "mean_cv_score": mean_score,
            "std_cv_score":  std_score,
            "params":        params,
            **{f"param_{k}": v for k, v in params.items()},
        })

    elapsed_total = time.perf_counter() - start

    # Salva resultados
    cv_results_df = pd.DataFrame(results).sort_values("mean_cv_score", ascending=False)
    cv_out = MODEL_DIR / "rf_cv_results.csv"
    cv_results_df.to_csv(cv_out, index=False)

    print()
    print("=" * 70)
    print("HYPERPARAMETER TUNING — RESUMO")
    print("=" * 70)
    print(f"Tempo total : {elapsed_total:.1f}s")
    print(f"Best {TUNING_SCORE_NAME} : {best_score:.4f}")
    print(f"Best params : {best_params}")
    print()

    pd.set_option("display.max_colwidth", None)
    top5 = cv_results_df.head(5)[["mean_cv_score", "std_cv_score", "params"]]
    print("Top 5 configurações:")
    print(top5.to_string(index=False))
    print()
    print(f"[SAVE] CV results → {cv_out}\n")

    return best_clf, cv_results_df

# ─── Evaluation ───────────────────────────────────────────────────────────────

def predict(clf, df: pd.DataFrame) -> np.ndarray:
    return clf.predict(build_feature_matrix(df))


def _print_report(y_true, y_pred, title: str):
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(classification_report(
        y_true, y_pred,
        target_names=["clean (0)", "bug-inducing (1)"],
        digits=4,
    ))
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual clean (0)", "Actual bug (1)"],
        columns=["Pred clean (0)", "Pred bug (1)"],
    )
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(cm_df.to_string())
    print()


def training_performance_report(clf, train_df: pd.DataFrame):
    X_train = build_feature_matrix(train_df)
    y_train = train_df["label"].values

    y_train_pred = clf.predict(X_train)
    _print_report(
        y_train, y_train_pred,
        f"TRAINING SET — IN-SAMPLE FIT (n={len(y_train):,})",
    )

    print(f"[CV] Out-of-fold predictions com {N_FOLDS}-fold StratifiedKFold …")
    cv    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_oof = cross_val_predict(clf, X_train, y_train, cv=cv, n_jobs=-1)
    _print_report(
        y_train, y_oof,
        f"TRAINING SET — {N_FOLDS}-FOLD OUT-OF-FOLD (honest estimate)",
    )


def test_performance_report(y_true, y_pred):
    _print_report(y_true, y_pred, "TEST SET CLASSIFICATION REPORT")


def feature_importance_report(clf):
    print("=" * 70)
    print("FEATURE IMPORTANCES (mean decrease in impurity)")
    print("=" * 70)
    feat_df = pd.DataFrame({
        "feature":    NUMERIC_FEATURES,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(feat_df.to_string(index=False))
    print()


def stratified_report(test_df: pd.DataFrame):
    """
    Tabela 1 — visão geral por reflete_mudanca (ambas as classes).
    Tabela 2 — foco na classe bug-inducing (label==1): acertos e erros
               por categoria de reflete_mudanca.
    """
    annotated = test_df[test_df["reflete_mudanca"].notna()].copy()
    missing   = test_df[test_df["reflete_mudanca"].isna()]

    print(f"[INFO] Commits com anotação reflete_mudanca  : {len(annotated):,}")
    print(f"[INFO] Commits sem anotação (excluídos)      : {len(missing):,}")
    print()

    if annotated.empty:
        print("[WARNING] Nenhum commit anotado — análise estratificada ignorada.")
        return

    # ── Tabela 1: visão geral ─────────────────────────────────────────────
    rows_geral = []
    for reflete_val in REFLETE_ORDER:
        stratum = annotated[annotated["reflete_mudanca"] == reflete_val]
        if stratum.empty:
            continue

        total   = len(stratum)
        errors  = int((stratum["pred"] != stratum["label"]).sum())
        correct = total - errors

        def stratum_recall(cls: int, _s=stratum) -> float:
            if not (_s["label"] == cls).any():
                return float("nan")
            return recall_score(
                _s["label"].values, _s["pred"].values,
                labels=[cls], average="macro", zero_division=0,
            )

        rows_geral.append({
            "reflete_mudanca": reflete_val,
            "total":           total,
            "acertos":         correct,
            "erros":           errors,
            "taxa_acerto":     round(correct / total, 4),
            "taxa_erro":       round(errors  / total, 4),
            "recall_clean":    round(stratum_recall(0), 4),
            "recall_bug":      round(stratum_recall(1), 4),
        })

    df_geral = pd.DataFrame(rows_geral)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("=" * 100)
    print("TABELA 1 — RESULTADO GERAL POR reflete_mudanca (ambas as classes)")
    print("=" * 100)
    print(df_geral.to_string(index=False))
    print()

    # ── Tabela 2: foco em bug-inducing (label == 1) ───────────────────────
    rows_bug = []
    for reflete_val in REFLETE_ORDER:
        stratum = annotated[
            (annotated["reflete_mudanca"] == reflete_val) &
            (annotated["label"] == 1)
        ]
        if stratum.empty:
            continue

        total        = len(stratum)
        # Acerto = predito como bug (1); Erro = predito como clean (0)
        acertos      = int((stratum["pred"] == 1).sum())
        erros        = int((stratum["pred"] == 0).sum())
        taxa_acerto  = acertos / total
        taxa_erro    = erros   / total

        rows_bug.append({
            "reflete_mudanca":  reflete_val,
            "total_bug":        total,
            "acertos (pred=1)": acertos,
            "erros   (pred=0)": erros,
            "taxa_acerto":      round(taxa_acerto, 4),
            "taxa_erro":        round(taxa_erro,   4),
        })

    df_bug = pd.DataFrame(rows_bug)

    print("=" * 100)
    print("TABELA 2 — DESEMPENHO NA CLASSE BUG-INDUCING (label=1) POR reflete_mudanca")
    print("           acerto = modelo prediz bug  |  erro = modelo prediz clean")
    print("=" * 100)
    print(df_bug.to_string(index=False))
    print()

    # ── Persiste ambas ────────────────────────────────────────────────────
    out_geral = MODEL_DIR / "stratified_results_rf_geral.csv"
    out_bug   = MODEL_DIR / "stratified_results_rf_bug.csv"
    df_geral.to_csv(out_geral, index=False)
    df_bug.to_csv(out_bug,   index=False)
    print(f"[INFO] Tabela geral salva → {out_geral}")
    print(f"[INFO] Tabela bug salva   → {out_bug}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("BUG-INDUCING COMMIT CLASSIFIER — RANDOM FOREST (10-fold CV tuning)")
    print("=" * 70)

    print("\n[LOAD] Carregando dados do diretório de treino …")
    full_df = load_all_data(TRAIN_DIR)
    full_df = full_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    total   = len(full_df)
    n_bug   = int((full_df["label"] == 1).sum())
    n_clean = int((full_df["label"] == 0).sum())
    print(f"\n[SPLIT] Total: {total:,}  (clean={n_clean:,}, bug={n_bug:,})")

    train_df, test_df = stratified_split(full_df, TEST_SIZE, RANDOM_STATE)
    print(f"[SPLIT] Train : {len(train_df):,}  "
          f"(clean={int((train_df['label']==0).sum()):,}, "
          f"bug={int((train_df['label']==1).sum()):,})")
    print(f"[SPLIT] Test  : {len(test_df):,}  "
          f"(clean={int((test_df['label']==0).sum()):,}, "
          f"bug={int((test_df['label']==1).sum()):,})\n")

    # ── Treino ────────────────────────────────────────────────────────────
    clf, _ = train_with_cv(train_df)

    model_path = MODEL_DIR / "rf_bug_classifier.joblib"
    dump(clf, model_path)
    print(f"[SAVE] Modelo salvo → {model_path}\n")

    # ── Avaliação ─────────────────────────────────────────────────────────
    training_performance_report(clf, train_df)

    test_df = test_df.copy()
    test_df["pred"] = predict(clf, test_df)
    test_performance_report(test_df["label"].values, test_df["pred"].values)

    feature_importance_report(clf)
    stratified_report(test_df)

    print("Done.")


if __name__ == "__main__":
    main()