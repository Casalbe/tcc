"""
train_rf.py — Bug-Inducing Commit Classifier (Random Forest)
==============================================================
- Split estratificado 70/30 a partir do diretório de treino
- Tuning de hiperparâmetros via 10-fold CV (grid search manual com progresso)
- Tabela estratificada por reflete_mudanca: acerto/erro na classe bug-inducing

Pode ser executado standalone (1 run, com progresso completo no terminal)
ou importado por run_experiments.py para repetição N vezes com seeds
diferentes, salvando métricas para análise estatística.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common import (
    NUMERIC_FEATURES,
    REFLETE_ORDER,
    TRAIN_DIR,
    build_feature_matrix,
    gmean_recall,
    load_all_data,
    shuffle_data,
    stratified_split,
)

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

N_FOLDS = 10

PARAM_GRID = {
    "n_estimators":      [100, 200],
    "max_depth":         [None, 3, 10, 21],
    "min_samples_split": [2, 10],
    "min_samples_leaf":  [1, 4],
    "max_features":      ["sqrt", "log2"],
}

TUNING_SCORE_NAME = "gmean_recall"
TUNING_SCORE      = make_scorer(gmean_recall)

# ─── Training com progresso em tempo real ─────────────────────────────────────

def train_with_cv(train_df: pd.DataFrame, seed: int, verbose: bool = True):
    """
    Itera manualmente sobre ParameterGrid para exibir o resultado de cada
    candidato assim que seus N_FOLDS folds terminam (quando verbose=True).

    `seed` controla tanto o StratifiedKFold quanto o RandomForestClassifier,
    para que cada run do experimento seja totalmente determinístico dado
    o seu seed.
    """
    X_train = build_feature_matrix(train_df)
    y_train = train_df["label"].values

    if verbose:
        print(f"[TRAIN] Feature matrix : {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
        print(f"[TRAIN] Distribuição   : "
              f"clean={int((y_train == 0).sum()):,}, bug={int((y_train == 1).sum()):,}\n")

    cv         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    candidates = list(ParameterGrid(PARAM_GRID))
    total      = len(candidates)
    total_fits = total * N_FOLDS

    if verbose:
        print(f"[TUNE] {total} candidatos × {N_FOLDS} folds = {total_fits} fits totais")
        print(f"[TUNE] Scoring: {TUNING_SCORE_NAME}  |  seed={seed}\n")
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
        clf = RandomForestClassifier(
            **params,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )

        scores = cross_val_score(
            clf, X_train, y_train,
            cv=cv,
            scoring=TUNING_SCORE,
            n_jobs=1,
        )

        mean_score = float(scores.mean())
        std_score  = float(scores.std())

        is_best = mean_score > best_score
        if is_best:
            best_score  = mean_score
            best_params = params
            best_clf = RandomForestClassifier(
                **params,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ).fit(X_train, y_train)

        if verbose:
            elapsed    = time.perf_counter() - start
            eta        = (elapsed / idx) * (total - idx)
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
    cv_results_df = pd.DataFrame(results).sort_values("mean_cv_score", ascending=False)

    if verbose:
        print()
        print("=" * 70)
        print("HYPERPARAMETER TUNING — RESUMO")
        print("=" * 70)
        print(f"Tempo total : {elapsed_total:.1f}s")
        print(f"Best {TUNING_SCORE_NAME} : {best_score:.4f}")
        print(f"Best params : {best_params}\n")

    return best_clf, cv_results_df, best_params, best_score

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


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Métricas-resumo de uma run, usadas na tabela final de 30 execuções."""
    metrics = {
        "accuracy":        accuracy_score(y_true, y_pred),
        "precision_bug":   precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_bug":      recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_bug":          f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_clean": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_clean":    recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_clean":        f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_macro":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "gmean_recall":    gmean_recall(y_true, y_pred),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")
    return metrics


def feature_importance_report(clf) -> pd.DataFrame:
    feat_df = pd.DataFrame({
        "feature":    NUMERIC_FEATURES,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    return feat_df


def stratified_report(test_df: pd.DataFrame, verbose: bool = True):
    """
    Tabela 1 — visão geral por reflete_mudanca (ambas as classes).
    Tabela 2 — foco na classe bug-inducing (label==1).
    Retorna ambos os DataFrames (mesmo quando verbose=False) para
    permitir agregação posterior nas 30 runs.
    """
    annotated = test_df[test_df["reflete_mudanca"].notna()].copy()
    missing   = test_df[test_df["reflete_mudanca"].isna()]

    if verbose:
        print(f"[INFO] Commits com anotação reflete_mudanca  : {len(annotated):,}")
        print(f"[INFO] Commits sem anotação (excluídos)      : {len(missing):,}\n")

    if annotated.empty:
        if verbose:
            print("[WARNING] Nenhum commit anotado — análise estratificada ignorada.")
        return pd.DataFrame(), pd.DataFrame()

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

    rows_bug = []
    for reflete_val in REFLETE_ORDER:
        stratum = annotated[
            (annotated["reflete_mudanca"] == reflete_val) &
            (annotated["label"] == 1)
        ]
        if stratum.empty:
            continue
        total       = len(stratum)
        acertos     = int((stratum["pred"] == 1).sum())
        erros       = int((stratum["pred"] == 0).sum())
        rows_bug.append({
            "reflete_mudanca":  reflete_val,
            "total_bug":        total,
            "acertos (pred=1)": acertos,
            "erros   (pred=0)": erros,
            "taxa_acerto":      round(acertos / total, 4),
            "taxa_erro":        round(erros   / total, 4),
        })
    df_bug = pd.DataFrame(rows_bug)

    if verbose:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 160)
        print("=" * 100)
        print("TABELA 1 — RESULTADO GERAL POR reflete_mudanca (ambas as classes)")
        print("=" * 100)
        print(df_geral.to_string(index=False))
        print()
        print("=" * 100)
        print("TABELA 2 — DESEMPENHO NA CLASSE BUG-INDUCING (label=1) POR reflete_mudanca")
        print("=" * 100)
        print(df_bug.to_string(index=False))
        print()

    return df_geral, df_bug

# ─── Single-seed experiment entry point (used by run_experiments.py) ─────────

def run_experiment(seed: int, verbose: bool = False, save_model: bool = False):
    """
    Executa um ciclo completo (load → shuffle → split → tune → eval) com
    um único seed controlando TUDO (shuffle do dataset, split train/test,
    StratifiedKFold do tuning, e o próprio RandomForestClassifier).

    Retorna um dict com as métricas de teste + metadados da run, pronto
    para ser empilhado em um DataFrame de 30 linhas (uma por run).
    """
    t0 = time.perf_counter()

    full_df = load_all_data(TRAIN_DIR)
    full_df = shuffle_data(full_df, seed)
    train_df, test_df = stratified_split(full_df, seed)

    clf, cv_results_df, best_params, best_cv_score = train_with_cv(
        train_df, seed, verbose=verbose
    )

    if save_model:
        dump(clf, MODEL_DIR / f"rf_seed{seed}.joblib")

    test_df = test_df.copy()
    test_df["pred"] = predict(clf, test_df)
    y_proba = clf.predict_proba(build_feature_matrix(test_df))[:, 1]

    metrics = compute_metrics(test_df["label"].values, test_df["pred"].values, y_proba)
    elapsed = time.perf_counter() - t0

    if verbose:
        test_performance_report_inline(test_df["label"].values, test_df["pred"].values)
        feat_df = feature_importance_report(clf)
        print(feat_df.to_string(index=False))
        print()
        stratified_report(test_df, verbose=True)

    return {
        "model":          "RandomForest",
        "seed":           seed,
        "n_train":        len(train_df),
        "n_test":         len(test_df),
        "best_params":    best_params,
        "best_cv_score":  best_cv_score,
        "elapsed_sec":    elapsed,
        **metrics,
    }


def test_performance_report_inline(y_true, y_pred):
    _print_report(y_true, y_pred, "TEST SET CLASSIFICATION REPORT")

# ─── Main (standalone single run, full verbosity) ─────────────────────────────

def main():
    print("=" * 70)
    print("BUG-INDUCING COMMIT CLASSIFIER — RANDOM FOREST (10-fold CV tuning)")
    print("=" * 70)

    seed = 42
    print(f"\n[LOAD] Carregando dados do diretório de treino (seed={seed}) …")
    result = run_experiment(seed=seed, verbose=True, save_model=True)

    print("=" * 70)
    print("RESUMO DA RUN")
    print("=" * 70)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("\nDone.")


if __name__ == "__main__":
    main()
