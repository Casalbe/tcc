"""
analyze_results.py — Análise estatística RF vs LR para a monografia
=======================================================================
Lê:
  - results/all_runs.csv         (1 linha por run: 30 RF + 30 LR)
  - results/summary_by_model.csv (1 linha por modelo: médias agregadas)

Gera, em results/figures/:
    01_overview_bar_means.png        — média ± desvio padrão das 6 métricas (summary CSV)
    02_boxplots_grid.png             — boxplots por métrica, RF vs LR (all_runs)
  03_class_comparison_rf.png       — bug vs clean dentro do RF (P/R/F1)
  04_class_comparison_lr.png       — bug vs clean dentro do LR (P/R/F1)
  05_class_comparison_grouped.png  — bug vs clean, RF e LR juntos
  06_paired_differences.png        — diferença pareada (RF-LR) por seed
    07_radar_comparison.png          — radar das 6 métricas, RF vs LR
    08_quality_prediction_link.png   — qualidade da mensagem vs saída do modelo
        09_reflection_class_metrics.png   — métricas por classe de reflexão

  comparison_report.pdf            — todas as figuras acima, em um PDF único
    statistical_tests.csv            — Wilcoxon pareado + teste t pareado por métrica
    statistical_tests.txt            — mesmo conteúdo, formatado para leitura/colar na tese
    quality_correlation.csv          — correlação da qualidade da mensagem com as predições
    quality_correlation.txt          — resumo textual da correlação
    reflection_class_metrics.csv     — precisão/recall/F1 por classe de reflexão
    reflection_class_metrics.txt     — resumo textual por classe de reflexão

Uso:
    python analyze_results.py
    python analyze_results.py --results-dir results --output-dir results/figures
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.metrics import f1_score, precision_score, recall_score

from common import normalize_reflete_mudanca, reflete_to_score

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

# Apenas estas 6 métricas entram na análise final; CV e accuracy foram removidos.
METRICS = [
    "precision_bug",
    "recall_bug",
    "f1_bug",
    "precision_clean",
    "recall_clean",
    "f1_clean",
]

METRIC_LABELS = {
    "precision_bug":    "Precision\n(bug-inducing)",
    "recall_bug":       "Recall\n(bug-inducing)",
    "f1_bug":           "F1-score\n(bug-inducing)",
    "precision_clean":  "Precision\n(clean)",
    "recall_clean":     "Recall\n(clean)",
    "f1_clean":         "F1-score\n(clean)",
}

MODEL_ORDER  = ["RandomForest", "LogisticRegression"]
MODEL_LABELS = {"RandomForest": "Random Forest", "LogisticRegression": "Logistic Regression"}
MODEL_COLORS = {"RandomForest": "#2E7D32", "LogisticRegression": "#1565C0"}

REFLETE_PLOT_ORDER = ["Não", "Provavelmente não", "Provavelmente sim", "Sim"]
REFLETE_COLORS = {
    "Não": "#B71C1C",
    "Provavelmente não": "#E53935",
    "Provavelmente sim": "#FB8C00",
    "Sim": "#2E7D32",
}

CLASS_METRICS = {
    "bug-inducing": ["precision_bug", "recall_bug", "f1_bug"],
    "clean":        ["precision_clean", "recall_clean", "f1_clean"],
}
CLASS_COLORS = {"bug-inducing": "#C62828", "clean": "#00897B"}

plt.rcParams.update({
    "figure.dpi":      120,
    "savefig.dpi":     150,
    "font.size":       10,
    "axes.titlesize":  12,
    "axes.titleweight": "bold",
    "axes.labelsize":  10,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(results_dir: Path):
    all_runs_path = results_dir / "all_runs.csv"
    summary_path  = results_dir / "summary_by_model.csv"
    predictions_path = results_dir / "test_predictions.csv"

    for p in (all_runs_path, summary_path):
        if not p.exists():
            raise SystemExit(f"[ERROR] Arquivo não encontrado: {p}")

    all_runs = pd.read_csv(all_runs_path, sep=",")
    summary  = pd.read_csv(summary_path, sep=",")
    predictions = pd.read_csv(predictions_path, sep=",") if predictions_path.exists() else None

    missing_in_runs = [m for m in METRICS if m not in all_runs.columns]
    if missing_in_runs:
        raise SystemExit(f"[ERROR] Colunas ausentes em all_runs.csv: {missing_in_runs}")

    mean_cols = [f"{m}_mean" for m in METRICS]
    std_cols = [f"{m}_std" for m in METRICS]
    missing_in_summary = [c for c in mean_cols + std_cols if c not in summary.columns]
    if missing_in_summary:
        raise SystemExit(f"[ERROR] Colunas ausentes em summary_by_model.csv: {missing_in_summary}")

    present_models = [m for m in MODEL_ORDER if m in all_runs["model"].unique()]
    if len(present_models) < 2:
        print(f"[WARNING] Esperava 2 modelos, encontrou: {present_models}")

    if predictions is None:
        print(f"[WARNING] Arquivo não encontrado: {predictions_path} — análise de reflete_mudanca será pulada.")

    return all_runs, summary, predictions, present_models


def get_paired_wide(all_runs: pd.DataFrame, metric: str, models) -> pd.DataFrame:
    """Pivota all_runs para ter uma linha por seed e uma coluna por modelo,
    mantendo apenas seeds presentes em AMBOS os modelos (pareamento)."""
    sub = all_runs[["model", "seed", metric]]
    wide = sub.pivot(index="seed", columns="model", values=metric)
    wide = wide.dropna(subset=models)
    return wide


def _prepare_prediction_log(predictions: pd.DataFrame) -> pd.DataFrame:
    df = predictions.copy()
    df["reflete_mudanca"] = df["reflete_mudanca"].map(normalize_reflete_mudanca)
    df["reflete_score"] = df["reflete_mudanca"].map(reflete_to_score)
    df = df[df["reflete_score"].notna()].copy()
    df["reflete_score"] = df["reflete_score"].astype(int)
    df["correct"] = df["correct"].astype(int)
    df["label"] = df["label"].astype(int)
    df["pred"] = df["pred"].astype(int)
    df["pred_bug_proba"] = df["pred_bug_proba"].astype(float)
    return df


def summarize_quality_correlations(predictions: pd.DataFrame, models) -> pd.DataFrame:
    rows = []

    for model in models:
        model_df = predictions[predictions["model"] == model].copy()
        if model_df.empty:
            continue

        scopes = {
            "all_annotated": model_df,
            "bug_inducing_only": model_df[model_df["label"] == 1],
        }

        for scope_name, scope_df in scopes.items():
            if scope_df.empty:
                continue

            per_seed_rows = []
            for seed, seed_df in scope_df.groupby("seed"):
                if seed_df["reflete_score"].nunique() < 2:
                    continue

                rho_prob, p_prob = stats.spearmanr(seed_df["reflete_score"], seed_df["pred_bug_proba"])
                rho_correct, p_correct = stats.spearmanr(seed_df["reflete_score"], seed_df["correct"])

                per_seed_rows.append({
                    "model": model,
                    "scope": scope_name,
                    "seed": seed,
                    "n_rows": len(seed_df),
                    "rho_quality_vs_pred_bug_proba": rho_prob,
                    "p_quality_vs_pred_bug_proba": p_prob,
                    "rho_quality_vs_correct": rho_correct,
                    "p_quality_vs_correct": p_correct,
                })

            if not per_seed_rows:
                continue

            per_seed = pd.DataFrame(per_seed_rows)
            rows.append({
                "model": model,
                "scope": scope_name,
                "n_seeds": len(per_seed),
                "n_rows": int(per_seed["n_rows"].sum()),
                "mean_rho_quality_vs_pred_bug_proba": per_seed["rho_quality_vs_pred_bug_proba"].mean(),
                "std_rho_quality_vs_pred_bug_proba": per_seed["rho_quality_vs_pred_bug_proba"].std(),
                "mean_p_quality_vs_pred_bug_proba": per_seed["p_quality_vs_pred_bug_proba"].mean(),
                "mean_rho_quality_vs_correct": per_seed["rho_quality_vs_correct"].mean(),
                "std_rho_quality_vs_correct": per_seed["rho_quality_vs_correct"].std(),
                "mean_p_quality_vs_correct": per_seed["p_quality_vs_correct"].mean(),
            })

    return pd.DataFrame(rows)


def summarize_quality_by_class(predictions: pd.DataFrame, models) -> pd.DataFrame:
    rows = []

    for model in models:
        model_df = predictions[(predictions["model"] == model) & (predictions["label"] == 1)].copy()
        if model_df.empty:
            continue

        for reflete_val in REFLETE_PLOT_ORDER:
            class_df = model_df[model_df["reflete_mudanca"] == reflete_val]
            if class_df.empty:
                continue

            per_seed = class_df.groupby("seed").agg(
                mean_pred_bug_proba=("pred_bug_proba", "mean"),
                false_negative_rate=("pred", lambda s: float((s == 0).mean())),
                n_rows=("pred", "size"),
            )

            rows.append({
                "model": model,
                "reflete_mudanca": reflete_val,
                "n_seeds": len(per_seed),
                "n_rows": int(per_seed["n_rows"].sum()),
                "mean_pred_bug_proba": per_seed["mean_pred_bug_proba"].mean(),
                "std_pred_bug_proba": per_seed["mean_pred_bug_proba"].std(),
                "mean_false_negative_rate": per_seed["false_negative_rate"].mean(),
                "std_false_negative_rate": per_seed["false_negative_rate"].std(),
            })

    return pd.DataFrame(rows)


def summarize_reflection_class_metrics(predictions: pd.DataFrame, models) -> pd.DataFrame:
    rows = []

    for model in models:
        model_df = predictions[predictions["model"] == model].copy()
        if model_df.empty:
            continue

        for reflete_val in REFLETE_PLOT_ORDER:
            class_df = model_df[model_df["reflete_mudanca"] == reflete_val]
            if class_df.empty:
                continue

            per_seed_rows = []
            for seed, seed_df in class_df.groupby("seed"):
                y_true = seed_df["label"].astype(int).values
                y_pred = seed_df["pred"].astype(int).values
                per_seed_rows.append({
                    "model": model,
                    "reflete_mudanca": reflete_val,
                    "seed": seed,
                    "n_rows": len(seed_df),
                    "precision_bug": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
                    "recall_bug": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
                    "f1_bug": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
                    "precision_clean": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
                    "recall_clean": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
                    "f1_clean": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
                    "bug_rate": float((y_true == 1).mean()),
                    "pred_bug_rate": float((y_pred == 1).mean()),
                })

            if not per_seed_rows:
                continue

            per_seed = pd.DataFrame(per_seed_rows)
            rows.append({
                "model": model,
                "reflete_mudanca": reflete_val,
                "n_seeds": len(per_seed),
                "n_rows": int(per_seed["n_rows"].sum()),
                "precision_bug_mean": per_seed["precision_bug"].mean(),
                "precision_bug_std": per_seed["precision_bug"].std(),
                "recall_bug_mean": per_seed["recall_bug"].mean(),
                "recall_bug_std": per_seed["recall_bug"].std(),
                "f1_bug_mean": per_seed["f1_bug"].mean(),
                "f1_bug_std": per_seed["f1_bug"].std(),
                "precision_clean_mean": per_seed["precision_clean"].mean(),
                "precision_clean_std": per_seed["precision_clean"].std(),
                "recall_clean_mean": per_seed["recall_clean"].mean(),
                "recall_clean_std": per_seed["recall_clean"].std(),
                "f1_clean_mean": per_seed["f1_clean"].mean(),
                "f1_clean_std": per_seed["f1_clean"].std(),
                "bug_rate_mean": per_seed["bug_rate"].mean(),
                "bug_rate_std": per_seed["bug_rate"].std(),
                "pred_bug_rate_mean": per_seed["pred_bug_rate"].mean(),
                "pred_bug_rate_std": per_seed["pred_bug_rate"].std(),
            })

    return pd.DataFrame(rows)


# ─── Figure 1: Overview bar chart (summary means) ─────────────────────────────

def fig_overview_bar_means(summary: pd.DataFrame, models, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(METRICS))
    width = 0.35

    for i, model in enumerate(models):
        row = summary[summary["model"] == model]
        if row.empty:
            continue
        values = [row[f"{m}_mean"].values[0] for m in METRICS]
        stds = [row[f"{m}_std"].values[0] for m in METRICS]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width, yerr=stds, capsize=4,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, None),
            edgecolor="white", linewidth=0.8,
        )
        for b, v, s in zip(bars, values, stds):
            ax.text(b.get_x() + b.get_width() / 2, v + s + 0.015, f"{v:.3f}±{s:.3f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=8.5)
    ax.set_ylabel("Score médio (30 runs)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Visão geral — média ± desvio padrão por métrica")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

    if standalone:
        fig.tight_layout()
        return fig
    return ax


# ─── Figure 2: Boxplots grid (per-run distributions) ──────────────────────────

def fig_boxplots_grid(all_runs: pd.DataFrame, models):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        data, labels, colors = [], [], []
        for model in models:
            vals = all_runs.loc[all_runs["model"] == model, metric].dropna().values
            data.append(vals)
            labels.append(MODEL_LABELS.get(model, model))
            colors.append(MODEL_COLORS.get(model, "#888888"))

        bp = ax.boxplot(
            data, labels=labels, patch_artist=True, widths=0.55,
            medianprops=dict(color="black", linewidth=1.5),
            flierprops=dict(marker="o", markersize=4, alpha=0.6),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)

        # jitter dos pontos individuais (cada run) sobre o boxplot
        rng = np.random.default_rng(42)
        for i, vals in enumerate(data, start=1):
            jitter = rng.normal(0, 0.05, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       s=10, color="black", alpha=0.35, zorder=3)

        ax.set_title(METRIC_LABELS[metric], fontsize=10)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.set_ylim(-0.02, 1.05)

    fig.suptitle(
        "Distribuição por run (30 execuções) — Random Forest vs Logistic Regression",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


# ─── Figures 3 & 4: Class comparison within a single model ───────────────────

def fig_class_comparison_single_model(all_runs: pd.DataFrame, model: str):
    fig, ax = plt.subplots(figsize=(8, 6))

    sub_metrics = ["precision", "recall", "f1"]
    x = np.arange(len(sub_metrics))
    width = 0.35

    for i, (cls, cols) in enumerate(CLASS_METRICS.items()):
        means = []
        stds = []
        for col in cols:
            vals = all_runs.loc[all_runs["model"] == model, col].dropna().values
            means.append(vals.mean())
            stds.append(vals.std())
        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset, means, width, yerr=stds, capsize=4,
            label=cls, color=CLASS_COLORS[cls], edgecolor="white", linewidth=0.8,
        )
        for b, v, s in zip(bars, means, stds):
            ax.text(b.get_x() + b.get_width() / 2, v + s + 0.03, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1-score"])
    ax.set_ylabel("Score médio ± desvio padrão (30 runs)")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{MODEL_LABELS.get(model, model)} — Bug-inducing vs Clean")
    ax.legend(title="Classe", loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)

    fig.tight_layout()
    return fig


# ─── Figure 5: Class comparison, both models grouped ──────────────────────────

def fig_class_comparison_grouped(all_runs: pd.DataFrame, models):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, (cls, cols) in zip(axes, CLASS_METRICS.items()):
        x = np.arange(len(cols))
        width = 0.35
        for i, model in enumerate(models):
            means, stds = [], []
            for col in cols:
                vals = all_runs.loc[all_runs["model"] == model, col].dropna().values
                means.append(vals.mean())
                stds.append(vals.std())
            offset = (i - (len(models) - 1) / 2) * width
            bars = ax.bar(
                x + offset, means, width, yerr=stds, capsize=4,
                label=MODEL_LABELS.get(model, model),
                color=MODEL_COLORS.get(model, None),
                edgecolor="white", linewidth=0.8,
            )
            for b, v, s in zip(bars, means, stds):
                ax.text(b.get_x() + b.get_width() / 2, v + s + 0.03, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels(["Precision", "Recall", "F1-score"])
        ax.set_title(f"Classe: {cls}")
        ax.set_ylim(0, 1.1)

    axes[0].set_ylabel("Score médio ± desvio padrão (30 runs)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Comparação por classe — Random Forest vs Logistic Regression", fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# ─── Figure 6: Paired differences per seed ────────────────────────────────────

def fig_paired_differences(all_runs: pd.DataFrame, models):
    if len(models) < 2:
        return None
    model_a, model_b = models[0], models[1]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, METRICS):
        wide = get_paired_wide(all_runs, metric, [model_a, model_b])
        if wide.empty:
            ax.set_visible(False)
            continue
        diff = wide[model_a] - wide[model_b]
        colors = ["#2E7D32" if d >= 0 else "#C62828" for d in diff]

        ax.bar(diff.index.astype(str), diff.values, color=colors, alpha=0.75, width=0.7)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(diff.mean(), color="black", linewidth=1.2, linestyle="--", alpha=0.6)
        ax.set_title(METRIC_LABELS[metric], fontsize=9.5)
        ax.set_xticks([])
        ax.set_xlabel(f"seeds (n={len(diff)})", fontsize=8)
        ax.text(
            0.02, 0.95, f"média={diff.mean():+.3f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"),
        )

    fig.suptitle(
        f"Diferença pareada por seed: {MODEL_LABELS.get(model_a, model_a)} − "
        f"{MODEL_LABELS.get(model_b, model_b)}\n"
        f"(barras verdes = RF melhor nesse seed; vermelhas = LR melhor)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ─── Figure 7: Radar chart ─────────────────────────────────────────────────────

def fig_radar(summary: pd.DataFrame, models):
    angles = np.linspace(0, 2 * np.pi, len(METRICS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for model in models:
        row = summary[summary["model"] == model]
        if row.empty:
            continue
        values = [row[f"{m}_mean"].values[0] for m in METRICS]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=MODEL_LABELS.get(model, model),
                color=MODEL_COLORS.get(model, None))
        ax.fill(angles, values, alpha=0.15, color=MODEL_COLORS.get(model, None))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([METRIC_LABELS[m].replace("\n", " ") for m in METRICS], fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_title("Comparação geral (radar) — médias das 6 métricas", pad=20, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False)

    fig.tight_layout()
    return fig


# ─── Figure 8: Commit-message quality vs prediction outcome ──────────────────

def fig_quality_prediction_link(quality_summary: pd.DataFrame, models):
    if quality_summary.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    metrics = [
        ("mean_pred_bug_proba", "std_pred_bug_proba", "Predicted bug probability"),
        ("mean_false_negative_rate", "std_false_negative_rate", "False negative rate"),
    ]

    for col_idx, model in enumerate(models):
        model_df = quality_summary[quality_summary["model"] == model]
        if model_df.empty:
            continue

        for row_idx, (mean_col, std_col, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            bars = []
            x = np.arange(len(REFLETE_PLOT_ORDER))
            values = []
            stds = []
            for reflete_val in REFLETE_PLOT_ORDER:
                row = model_df[model_df["reflete_mudanca"] == reflete_val]
                if row.empty:
                    values.append(np.nan)
                    stds.append(np.nan)
                else:
                    values.append(row[mean_col].values[0])
                    stds.append(row[std_col].values[0])

            values_arr = np.array(values, dtype=float)
            stds_arr = np.array(stds, dtype=float)
            stds_arr = np.nan_to_num(stds_arr, nan=0.0)
            bars = ax.bar(
                x,
                values_arr,
                yerr=stds_arr,
                capsize=4,
                color=[REFLETE_COLORS[label] for label in REFLETE_PLOT_ORDER],
                edgecolor="white",
                linewidth=0.8,
            )
            for b, v, s in zip(bars, values_arr, stds_arr):
                if np.isnan(v):
                    continue
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + (0.015 if row_idx == 0 else 0.02),
                    f"{v:.3f}±{(0.0 if np.isnan(s) else s):.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

            ax.set_ylim(0, 1.05)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{MODEL_LABELS.get(model, model)}")
            ax.grid(axis="y", alpha=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels(REFLETE_PLOT_ORDER, rotation=15)

    axes[0, 0].set_title(f"{MODEL_LABELS.get(models[0], models[0])} — defect-inducing commits")
    if len(models) > 1:
        axes[0, 1].set_title(f"{MODEL_LABELS.get(models[1], models[1])} — defect-inducing commits")
    fig.suptitle(
        "Commit-message quality vs predicted bug outcome\n"
        "Defect-inducing commits only; bars show mean across seeds ± std of seed means",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ─── Figure 9: Reflection-class performance comparison ──────────────────────

def fig_reflection_class_metrics(reflection_metrics: pd.DataFrame, models):
    if reflection_metrics.empty:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.flatten()
    metric_specs = [
        ("precision_bug_mean", "precision_bug_std", "Precision (bug-inducing)"),
        ("recall_bug_mean", "recall_bug_std", "Recall (bug-inducing)"),
        ("f1_bug_mean", "f1_bug_std", "F1-score (bug-inducing)"),
        ("precision_clean_mean", "precision_clean_std", "Precision (clean)"),
        ("recall_clean_mean", "recall_clean_std", "Recall (clean)"),
        ("f1_clean_mean", "f1_clean_std", "F1-score (clean)"),
    ]

    x = np.arange(len(REFLETE_PLOT_ORDER))
    width = 0.35

    for ax, (mean_col, std_col, title) in zip(axes, metric_specs):
        for i, model in enumerate(models):
            model_df = reflection_metrics[reflection_metrics["model"] == model]
            means = []
            stds = []
            for reflete_val in REFLETE_PLOT_ORDER:
                row = model_df[model_df["reflete_mudanca"] == reflete_val]
                if row.empty:
                    means.append(np.nan)
                    stds.append(0.0)
                else:
                    means.append(row[mean_col].values[0])
                    stds.append(0.0 if pd.isna(row[std_col].values[0]) else row[std_col].values[0])

            offset = (i - (len(models) - 1) / 2) * width
            bars = ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=4,
                label=MODEL_LABELS.get(model, model),
                color=MODEL_COLORS.get(model, None),
                edgecolor="white",
                linewidth=0.8,
            )
            for b, v, s in zip(bars, means, stds):
                if pd.isna(v):
                    continue
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + s + 0.015,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

        ax.set_title(title)
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(REFLETE_PLOT_ORDER, rotation=15)
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)
    fig.suptitle(
        "Performance by reflection class — bug-inducing and clean metrics",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ─── Statistical tests ─────────────────────────────────────────────────────────

def run_statistical_tests(all_runs: pd.DataFrame, models) -> pd.DataFrame:
    """
    Teste pareado por seed (RF e LR compartilham o split treino/teste em
    cada seed, já que ambos usam o mesmo seed em run_experiments.py).
    - Wilcoxon signed-rank: não-paramétrico, recomendado para n=30 e sem
      garantia de normalidade — é o teste padrão para comparar 2 modelos
      em K runs pareados na literatura de SE empírica.
    - Teste t pareado: incluído como referência paramétrica complementar.
    """
    if len(models) < 2:
        print("[WARNING] Menos de 2 modelos — testes estatísticos pareados ignorados.")
        return pd.DataFrame()

    model_a, model_b = models[0], models[1]
    rows = []

    for metric in METRICS:
        wide = get_paired_wide(all_runs, metric, [model_a, model_b])
        n = len(wide)
        if n < 2:
            continue

        a = wide[model_a].values
        b = wide[model_b].values
        diff = a - b

        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(a, b)
        except ValueError:
            wilcoxon_stat, wilcoxon_p = np.nan, np.nan

        ttest_stat, ttest_p = stats.ttest_rel(a, b)

        rows.append({
            "metric":            metric,
            "n_paired_seeds":    n,
            f"mean_{model_a}":   a.mean(),
            f"std_{model_a}":    a.std(),
            f"mean_{model_b}":   b.mean(),
            f"std_{model_b}":    b.std(),
            "mean_diff_A_minus_B": diff.mean(),
            "std_diff":          diff.std(),
            "wilcoxon_stat":     wilcoxon_stat,
            "wilcoxon_p":        wilcoxon_p,
            "ttest_stat":        ttest_stat,
            "ttest_p":           ttest_p,
            "significant_0.05":  (wilcoxon_p < 0.05) if not np.isnan(wilcoxon_p) else None,
        })

    return pd.DataFrame(rows)


def write_text_report(test_df: pd.DataFrame, models, out_path: Path):
    model_a, model_b = (models + [None, None])[:2]
    lines = []
    lines.append("=" * 78)
    lines.append("TESTES ESTATÍSTICOS PAREADOS — Random Forest vs Logistic Regression")
    lines.append("=" * 78)
    lines.append(f"Comparação: {MODEL_LABELS.get(model_a, model_a)} (A) vs "
                  f"{MODEL_LABELS.get(model_b, model_b)} (B)")
    lines.append("Pareamento: mesmo seed → mesmo split treino/teste em ambos os modelos.")
    lines.append("H0 (Wilcoxon/teste t): não há diferença sistemática entre A e B na métrica.")
    lines.append("")

    for _, row in test_df.iterrows():
        sig = "SIM (p < 0.05)" if row["significant_0.05"] else "não"
        lines.append(f"--- {row['metric']} ---")
        lines.append(f"  n pares (seeds)      : {int(row['n_paired_seeds'])}")
        lines.append(f"  A ({model_a:<19s}): {row[f'mean_{model_a}']:.4f} ± {row[f'std_{model_a}']:.4f}")
        lines.append(f"  B ({model_b:<19s}): {row[f'mean_{model_b}']:.4f} ± {row[f'std_{model_b}']:.4f}")
        lines.append(f"  diferença média (A-B): {row['mean_diff_A_minus_B']:+.4f} "
                      f"± {row['std_diff']:.4f}")
        lines.append(f"  Wilcoxon signed-rank : stat={row['wilcoxon_stat']:.3f}  p={row['wilcoxon_p']:.4f}")
        lines.append(f"  Teste t pareado      : stat={row['ttest_stat']:.3f}  p={row['ttest_p']:.4f}")
        lines.append(f"  Diferença significativa a 5%? {sig}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_quality_report(
    corr_df: pd.DataFrame,
    class_df: pd.DataFrame,
    models,
    out_path: Path,
):
    lines = []
    lines.append("=" * 78)
    lines.append("QUALIDADE DA MENSAGEM DE COMMIT VS SAÍDA DO MODELO")
    lines.append("=" * 78)
    lines.append("Pontuação ordinal da qualidade: Não=0, Provavelmente não=1, Provavelmente sim=2, Sim=3.")
    lines.append("A análise principal usa commits defect-inducing e resume a variação por seed.")
    lines.append("")

    for model in models:
        lines.append(f"--- {MODEL_LABELS.get(model, model)} ---")
        for scope in ("all_annotated", "bug_inducing_only"):
            row = corr_df[(corr_df["model"] == model) & (corr_df["scope"] == scope)]
            if row.empty:
                continue
            row = row.iloc[0]
            lines.append(
                f"  {scope}: rho(qualidade, prob_bug) = {row['mean_rho_quality_vs_pred_bug_proba']:+.4f} "
                f"± {row['std_rho_quality_vs_pred_bug_proba']:.4f} | "
                f"rho(qualidade, acerto) = {row['mean_rho_quality_vs_correct']:+.4f} "
                f"± {row['std_rho_quality_vs_correct']:.4f}"
            )

        quality_rows = class_df[class_df["model"] == model]
        if not quality_rows.empty:
            lines.append("  Por classe de reflexão (somente defect-inducing):")
            for reflete_val in REFLETE_PLOT_ORDER:
                row = quality_rows[quality_rows["reflete_mudanca"] == reflete_val]
                if row.empty:
                    continue
                row = row.iloc[0]
                lines.append(
                    f"    {reflete_val:<18s} prob_bug={row['mean_pred_bug_proba']:.4f} ± {row['std_pred_bug_proba']:.4f}"
                    f" | fnr={row['mean_false_negative_rate']:.4f} ± {row['std_false_negative_rate']:.4f}"
                )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_reflection_class_report(reflection_df: pd.DataFrame, models, out_path: Path):
    lines = []
    lines.append("=" * 78)
    lines.append("PERFORMANCE POR CLASSE DE REFLEXÃO")
    lines.append("=" * 78)
    lines.append("Cada linha resume média ± desvio padrão ao longo dos seeds.")
    lines.append("")

    for model in models:
        lines.append(f"--- {MODEL_LABELS.get(model, model)} ---")
        model_df = reflection_df[reflection_df["model"] == model]
        for reflete_val in REFLETE_PLOT_ORDER:
            row = model_df[model_df["reflete_mudanca"] == reflete_val]
            if row.empty:
                continue
            row = row.iloc[0]
            lines.append(
                f"  {reflete_val:<18s} "
                f"Pbug={row['precision_bug_mean']:.4f} ± {row['precision_bug_std']:.4f} | "
                f"Rbug={row['recall_bug_mean']:.4f} ± {row['recall_bug_std']:.4f} | "
                f"F1bug={row['f1_bug_mean']:.4f} ± {row['f1_bug_std']:.4f}"
            )
            lines.append(
                f"{'':22s} "
                f"Pclean={row['precision_clean_mean']:.4f} ± {row['precision_clean_std']:.4f} | "
                f"Rclean={row['recall_clean_mean']:.4f} ± {row['recall_clean_std']:.4f} | "
                f"F1clean={row['f1_clean_mean']:.4f} ± {row['f1_clean_std']:.4f}"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Análise estatística RF vs LR.")
    parser.add_argument("--results-dir", type=str, default="results",
                         help="Diretório contendo all_runs.csv e summary_by_model.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                         help="Diretório de saída para figuras (default: <results-dir>/figures)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ANÁLISE ESTATÍSTICA — Random Forest vs Logistic Regression")
    print("=" * 70)

    all_runs, summary, predictions, models = load_data(results_dir)
    print(f"[LOAD] all_runs.csv          : {len(all_runs)} linhas")
    print(f"[LOAD] summary_by_model.csv  : {len(summary)} linhas")
    print(f"[LOAD] Modelos detectados    : {models}\n")

    figures = []

    print("[FIG] 01 — overview bar (médias)")
    figures.append(("01_overview_bar_means", fig_overview_bar_means(summary, models)))

    print("[FIG] 02 — boxplots grid")
    figures.append(("02_boxplots_grid", fig_boxplots_grid(all_runs, models)))

    for model in models:
        idx = "03" if model == models[0] else "04"
        print(f"[FIG] {idx} — class comparison ({model})")
        figures.append((f"{idx}_class_comparison_{model.lower()}",
                         fig_class_comparison_single_model(all_runs, model)))

    print("[FIG] 05 — class comparison grouped")
    figures.append(("05_class_comparison_grouped", fig_class_comparison_grouped(all_runs, models)))

    print("[FIG] 06 — paired differences per seed")
    fig6 = fig_paired_differences(all_runs, models)
    if fig6 is not None:
        figures.append(("06_paired_differences", fig6))

    print("[FIG] 07 — radar comparison")
    figures.append(("07_radar_comparison", fig_radar(summary, models)))

    quality_corr_df = pd.DataFrame()
    quality_class_df = pd.DataFrame()
    reflection_class_df = pd.DataFrame()

    if predictions is not None and not predictions.empty:
        print("[FIG] 08 — commit-message quality vs predicted outcomes")
        prepared_predictions = _prepare_prediction_log(predictions)
        quality_corr_df = summarize_quality_correlations(prepared_predictions, models)
        quality_class_df = summarize_quality_by_class(prepared_predictions, models)
        reflection_class_df = summarize_reflection_class_metrics(prepared_predictions, models)

        fig8 = fig_quality_prediction_link(quality_class_df, models)
        if fig8 is not None:
            figures.append(("08_quality_prediction_link", fig8))

        fig9 = fig_reflection_class_metrics(reflection_class_df, models)
        if fig9 is not None:
            figures.append(("09_reflection_class_metrics", fig9))

    # ── Salva PNGs individuais ────────────────────────────────────────────
    for name, fig in figures:
        png_path = output_dir / f"{name}.png"
        fig.savefig(png_path, bbox_inches="tight")
        print(f"  [SAVE] {png_path}")

    # ── Salva PDF combinado ────────────────────────────────────────────────
    pdf_path = output_dir / "comparison_report.pdf"
    with PdfPages(pdf_path) as pdf:
        for _, fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
    print(f"  [SAVE] {pdf_path}")

    for _, fig in figures:
        plt.close(fig)

    # ── Testes estatísticos ────────────────────────────────────────────────
    print("\n[STATS] Rodando testes pareados (Wilcoxon + teste t) por métrica …")
    test_df = run_statistical_tests(all_runs, models)
    if not test_df.empty:
        csv_path = output_dir / "statistical_tests.csv"
        txt_path = output_dir / "statistical_tests.txt"
        test_df.to_csv(csv_path, index=False)
        write_text_report(test_df, models, txt_path)
        print(f"  [SAVE] {csv_path}")
        print(f"  [SAVE] {txt_path}")

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", "{:.4f}".format)
        print("\nResumo dos testes:")
        print(test_df[["metric", "mean_diff_A_minus_B", "wilcoxon_p", "significant_0.05"]]
              .to_string(index=False))

    if not quality_corr_df.empty and not quality_class_df.empty:
        corr_csv = output_dir / "quality_correlation.csv"
        corr_txt = output_dir / "quality_correlation.txt"
        class_csv = output_dir / "quality_by_reflection.csv"
        quality_corr_df.to_csv(corr_csv, index=False)
        quality_class_df.to_csv(class_csv, index=False)
        write_quality_report(quality_corr_df, quality_class_df, models, corr_txt)
        print(f"  [SAVE] {corr_csv}")
        print(f"  [SAVE] {class_csv}")
        print(f"  [SAVE] {corr_txt}")

    if not reflection_class_df.empty:
        reflection_csv = output_dir / "reflection_class_metrics.csv"
        reflection_txt = output_dir / "reflection_class_metrics.txt"
        reflection_class_df.to_csv(reflection_csv, index=False)
        write_reflection_class_report(reflection_class_df, models, reflection_txt)
        print(f"  [SAVE] {reflection_csv}")
        print(f"  [SAVE] {reflection_txt}")

    print(f"\nDone. Resultados em: {output_dir}")


if __name__ == "__main__":
    main()