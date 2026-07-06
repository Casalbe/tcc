"""
run_experiments.py — Executa RF e LR N vezes cada para análise estatística
=============================================================================
Para cada um dos N_RUNS seeds independentes:
  1. Treina e avalia o Random Forest        (train_rf.run_experiment)
  2. Treina e avalia a Logistic Regression   (train_lr.run_experiment)

Cada seed controla TUDO naquela run (shuffle dos dados, split 70/30,
StratifiedKFold do tuning, e o próprio modelo) — então RF e LR na mesma
run usam o MESMO split treino/teste, o que torna o teste estatístico
pareado (ex.: Wilcoxon signed-rank, teste t pareado) o mais apropriado
para comparar os dois modelos.

Os resultados são salvos INCREMENTALMENTE (append a cada run finalizada)
em results/all_runs.csv, então o script pode ser interrompido e retomado
sem perder progresso já computado — basta rodar de novo, runs já presentes
no CSV são puladas.

Uso:
    python run_experiments.py                  # 30 runs de cada (default)
    python run_experiments.py --n-runs 50       # 50 runs de cada
    python run_experiments.py --only rf         # só Random Forest
    python run_experiments.py --only lr         # só Logistic Regression
    python run_experiments.py --seeds-from 1000 # seeds começam em 1000
"""

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import train_lr
import train_rf

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RESULTS_CSV = RESULTS_DIR / "all_runs.csv"
PREDICTIONS_CSV = RESULTS_DIR / "test_predictions.csv"
ERROR_LOG   = RESULTS_DIR / "errors.log"

DEFAULT_N_RUNS    = 30
DEFAULT_SEED_BASE = 1  # seeds = SEED_BASE, SEED_BASE+1, ..., SEED_BASE+N_RUNS-1

# Colunas fixas no início do CSV, na ordem em que aparecem; o resto das
# métricas (que vem de compute_metrics) é appended dinamicamente.
KEY_COLUMNS = ["model", "seed", "n_train", "n_test", "elapsed_sec"]


def load_existing_results() -> pd.DataFrame:
    if RESULTS_CSV.exists():
        return pd.read_csv(RESULTS_CSV)
    return pd.DataFrame(columns=KEY_COLUMNS)


def already_done(existing: pd.DataFrame, model: str, seed: int) -> bool:
    if existing.empty:
        return False
    return bool(((existing["model"] == model) & (existing["seed"] == seed)).any())


def append_result(result: dict):
    """Append de uma única linha ao CSV, criando o cabeçalho se necessário."""
    row_df = pd.DataFrame([result])
    # best_params é um dict -> serializa como string para o CSV
    if "best_params" in row_df.columns:
        row_df["best_params"] = row_df["best_params"].astype(str)

    write_header = not RESULTS_CSV.exists()
    if not write_header:
        existing_cols = list(pd.read_csv(RESULTS_CSV, nrows=0).columns)
        row_df = row_df.reindex(columns=existing_cols)
    row_df.to_csv(RESULTS_CSV, mode="a", header=write_header, index=False)


def log_error(model: str, seed: int, exc: Exception):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*70}\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] model={model} seed={seed}\n")
        f.write(traceback.format_exc())
    print(f"  [ERROR] {model} seed={seed} falhou — ver {ERROR_LOG}")


def run_all(n_runs: int, seed_base: int, only: str | None):
    seeds = list(range(seed_base, seed_base + n_runs))
    models_to_run = []
    if only in (None, "rf"):
        models_to_run.append(("RandomForest", train_rf))
    if only in (None, "lr"):
        models_to_run.append(("LogisticRegression", train_lr))

    existing   = load_existing_results()
    total_jobs = len(seeds) * len(models_to_run)
    job_idx    = 0

    print("=" * 70)
    print("EXPERIMENT RUNNER — RandomForest vs LogisticRegression")
    print("=" * 70)
    print(f"Runs por modelo : {n_runs}")
    print(f"Seeds           : {seeds[0]} … {seeds[-1]}")
    print(f"Modelos         : {[m for m, _ in models_to_run]}")
    print(f"Resultados →    : {RESULTS_CSV}")
    print(f"Total de jobs   : {total_jobs}\n")

    overall_start = time.perf_counter()

    for seed in seeds:
        for model_name, module in models_to_run:
            job_idx += 1
            tag = f"[{job_idx:>3}/{total_jobs}] {model_name:<19} seed={seed}"

            if already_done(existing, model_name, seed):
                print(f"{tag}  ⏭  já existe no CSV, pulando")
                continue

            print(f"{tag}  ▶ rodando …", flush=True)
            t0 = time.perf_counter()
            try:
                result = module.run_experiment(
                    seed=seed,
                    verbose=False,
                    save_model=False,
                    prediction_log_path=PREDICTIONS_CSV,
                )
                elapsed = time.perf_counter() - t0
                append_result(result)
                # Mantém o DataFrame em memória atualizado para já-feitos
                existing = pd.concat([existing, pd.DataFrame([result])], ignore_index=True)

                print(
                    f"{tag}  ✓ {elapsed:6.1f}s"
                    f"  f1_bug={result['f1_bug']:.4f}"
                    f"  recall_bug={result['recall_bug']:.4f}"
                    f"  gmean={result['gmean_recall']:.4f}"
                )
            except Exception as exc:  # noqa: BLE001 — queremos seguir rodando mesmo se uma run falhar
                log_error(model_name, seed, exc)
                continue

    overall_elapsed = time.perf_counter() - overall_start
    print(f"\nTempo total: {overall_elapsed/60:.1f} min")
    print(f"Resultados salvos em: {RESULTS_CSV}")

    summarize()


def summarize():
    """Imprime um resumo estatístico (média ± desvio padrão) por modelo,
    pronto para colar na seção de resultados da monografia."""
    if not RESULTS_CSV.exists():
        print("[INFO] Nenhum resultado ainda.")
        return

    df = pd.read_csv(RESULTS_CSV)
    metric_cols = [
        c for c in df.columns
        if c not in (
            "model",
            "seed",
            "n_train",
            "n_test",
            "best_params",
            "elapsed_sec",
            "accuracy",
            "best_cv_score",
        )
    ]

    print("\n" + "=" * 90)
    print("RESUMO ESTATÍSTICO POR MODELO (média ± desvio padrão sobre as runs)")
    print("=" * 90)

    summary_rows = []
    for model_name, group in df.groupby("model"):
        row = {"model": model_name, "n_runs": len(group)}
        for col in metric_cols:
            if pd.api.types.is_numeric_dtype(group[col]):
                row[f"{col}_mean"] = group[col].mean()
                row[f"{col}_std"]  = group[col].std()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(summary_df.to_string(index=False))

    summary_path = RESULTS_DIR / "summary_by_model.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SAVE] Resumo → {summary_path}")
    print(f"[SAVE] Dados completos (uma linha por run) → {RESULTS_CSV}")
    print("\nPara testes estatísticos pareados (Wilcoxon, teste t), una as runs")
    print("de RF e LR pelo mesmo `seed` — cada seed usa o mesmo split treino/teste.")


def main():
    parser = argparse.ArgumentParser(description="Roda RF e LR N vezes para análise estatística.")
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS,
                         help=f"Número de runs por modelo (default: {DEFAULT_N_RUNS})")
    parser.add_argument("--seeds-from", type=int, default=DEFAULT_SEED_BASE,
                         help=f"Primeiro seed da sequência (default: {DEFAULT_SEED_BASE})")
    parser.add_argument("--only", choices=["rf", "lr"], default=None,
                         help="Roda apenas um dos dois modelos")
    parser.add_argument("--summarize-only", action="store_true",
                         help="Não roda nada — apenas reimprime o resumo do CSV existente")
    args = parser.parse_args()

    if args.summarize_only:
        summarize()
        return

    run_all(n_runs=args.n_runs, seed_base=args.seeds_from, only=args.only)


if __name__ == "__main__":
    main()
