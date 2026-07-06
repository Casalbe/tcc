"""
train_lr.py — Bug-Inducing Commit Classifier (Logistic Regression)
=====================================================================
Equivalente linear ao train_rf.py — mesma lógica de dados (common.py),
mesmo split estratificado 70/30, mesmo esquema de tuning via 10-fold CV
e mesmas métricas de avaliação, para permitir comparação estatística
direta entre os dois modelos.

Diferenças específicas da Logistic Regression:
- Features são padronizadas (StandardScaler) antes do ajuste — regressão
  logística é sensível à escala das features, Random Forest não é.
- Grid de hiperparâmetros é o do `C` (inverso da força de regularização),
  penalidade e solver, em vez de profundidade/folhas de árvores.

Pode ser executado standalone (1 run, com progresso completo no terminal)
ou importado por run_experiments.py para repetição N vezes com seeds
diferentes.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common import (
    append_prediction_log,
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
# filterwarnings("ignore") nem sempre é repassado para os processos workers
# que joblib cria com n_jobs=-1 (cada subprocesso tem seu próprio estado de
# warnings). Registrar o filtro explicitamente para ConvergenceWarning aqui,
# no nível do módulo, garante que ele seja herdado quando os workers importam
# este arquivo.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

N_FOLDS = 10

# Grid equivalente em espírito ao PARAM_GRID do RF: varia a força de
# regularização (C), o tipo de penalidade e o solver compatível com cada uma.
PARAM_GRID = [
    {"C": [0.01, 0.1, 1.0, 10.0, 100.0], "penalty": ["l2"], "solver": ["lbfgs"]},
    {"C": [0.01, 0.1, 1.0, 10.0, 100.0], "penalty": ["l1"], "solver": ["liblinear"]},
    {"C": [0.01, 0.1, 1.0, 10.0, 100.0], "penalty": ["elasticnet"], "solver": ["saga"], "l1_ratio": [0.5]},
]

TUNING_SCORE_NAME = "gmean_recall"
TUNING_SCORE      = make_scorer(gmean_recall)

MAX_ITER = 5000

# ─── Training com progresso em tempo real ─────────────────────────────────────

def _make_pipeline(params: dict, seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            **params,
            class_weight="balanced",
            max_iter=MAX_ITER,
            random_state=seed,
        )),
    ])


def _fit_silently(pipe: Pipeline, X, y) -> Pipeline:
    """Fit suprimindo ConvergenceWarning explicitamente neste escopo —
    mais robusto do que apenas o filtro global, que pode não ser herdado
    pelos processos workers do joblib (n_jobs=-1)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        return pipe.fit(X, y)


def train_with_cv(train_df: pd.DataFrame, seed: int, verbose: bool = True):
    """
    Mesma estrutura de train_with_cv do RF: itera manualmente sobre
    ParameterGrid, mostrando progresso candidato a candidato.
    `seed` controla o StratifiedKFold e a LogisticRegression (via solver
    estocástico 'saga'; 'lbfgs'/'liblinear' são determinísticos mas o
    seed é passado de qualquer forma por consistência de API).
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
        pipe = _make_pipeline(params, seed)

        scores = cross_val_score(
            pipe, X_train, y_train,
            cv=cv,
            scoring=TUNING_SCORE,
            n_jobs=-1,
        )

        mean_score = float(scores.mean())
        std_score  = float(scores.std())

        is_best = mean_score > best_score
        if is_best:
            best_score  = mean_score
            best_params = params
            best_clf = _fit_silently(_make_pipeline(params, seed), X_train, y_train)

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
    metrics = {
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


def coefficient_report(clf) -> pd.DataFrame:
    """Equivalente à feature_importance_report do RF: coeficientes da LR
    (no espaço padronizado, já que o pipeline inclui o StandardScaler)."""
    coefs = clf.named_steps["clf"].coef_[0]
    feat_df = pd.DataFrame({
        "feature":    NUMERIC_FEATURES,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False)
    return feat_df


def stratified_report(test_df: pd.DataFrame, verbose: bool = True):
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

def run_experiment(
    seed: int,
    verbose: bool = False,
    save_model: bool = False,
    prediction_log_path: Path | None = None,
):
    """
    Mesma assinatura e mesmo contrato de retorno que train_rf.run_experiment,
    para que run_experiments.py trate os dois modelos de forma intercambiável.
    """
    t0 = time.perf_counter()

    full_df = load_all_data(TRAIN_DIR)
    full_df = shuffle_data(full_df, seed)
    train_df, test_df = stratified_split(full_df, seed)

    clf, cv_results_df, best_params, best_cv_score = train_with_cv(
        train_df, seed, verbose=verbose
    )

    if save_model:
        dump(clf, MODEL_DIR / f"lr_seed{seed}.joblib")

    test_df = test_df.copy()
    test_df["pred"] = predict(clf, test_df)
    y_proba = clf.predict_proba(build_feature_matrix(test_df))[:, 1]

    metrics = compute_metrics(test_df["label"].values, test_df["pred"].values, y_proba)
    elapsed = time.perf_counter() - t0

    if prediction_log_path is not None:
        append_prediction_log(test_df, "LogisticRegression", seed, y_proba, prediction_log_path)

    if verbose:
        _print_report(test_df["label"].values, test_df["pred"].values, "TEST SET CLASSIFICATION REPORT")
        coef_df = coefficient_report(clf)
        print(coef_df.to_string(index=False))
        print()
        stratified_report(test_df, verbose=True)

    return {
        "model":          "LogisticRegression",
        "seed":           seed,
        "n_train":        len(train_df),
        "n_test":         len(test_df),
        "best_params":    best_params,
        "elapsed_sec":    elapsed,
        **metrics,
    }

# ─── Main (standalone single run, full verbosity) ─────────────────────────────

def main():
    print("=" * 70)
    print("BUG-INDUCING COMMIT CLASSIFIER — LOGISTIC REGRESSION (10-fold CV tuning)")
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