#!/usr/bin/env python3
"""Comparação estatística PatchTST V5 vs TODAS as baselines (TCN, LSTM, GRU).

Generalização de `run_wilcoxon.py` (que compara só TCN vs V5) para rodar a
mesma comparação Wilcoxon signed-rank contra qualquer subconjunto das
baselines V1-V4 disponíveis, numa tabela só. Lê os `*_runs.csv` já
produzidos por:

    run_tcn_5seeds_fixed.py   -> ModelosNew/TCN/exp_tcn_5seeds_fixed/tcn_runs.csv
    run_lstm_5seeds.py        -> ModelosNew/LSTM/exp_lstm_5seeds/lstm_runs.csv
    run_gru_5seeds.py         -> ModelosNew/GRU/exp_gru_5seeds/gru_runs.csv

contra o melhor setup do V5 (default: exp_warmup = seq_len=16 + OneCycleLR,
ver docs/PLANO_TRANSFORMER_V5.md Secao 0.3).

Baselines cujo CSV ainda não existe são puladas com um aviso (não é erro —
permite rodar a comparação parcial enquanto LSTM/GRU ainda estão treinando).

Uso:
    cd Horus-CDS && python scripts/run_wilcoxon_baselines.py
    cd Horus-CDS && python scripts/run_wilcoxon_baselines.py --v5-exp exp_optimal
    cd Horus-CDS && python scripts/run_wilcoxon_baselines.py --baselines tcn,lstm
"""

import argparse, os, sys

PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINUX_DIR       = os.path.join(PROJECT_ROOT, "root", "Linux")
MODELOS_DIR     = os.path.join(LINUX_DIR, "DadosDoPostreino", "ModelosNew")
TRANSFORMER_DIR = os.path.join(MODELOS_DIR, "Transformer")

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ── Baselines conhecidas ────────────────────────────────────────────────────
# name -> (csv_path, label)
BASELINES = {
    'tcn': (
        os.path.join(MODELOS_DIR, "TCN", "exp_tcn_5seeds_fixed", "tcn_runs.csv"),
        "TCN V4 (exp_tcn_5seeds_fixed)",
    ),
    'lstm': (
        os.path.join(MODELOS_DIR, "LSTM", "exp_lstm_5seeds", "lstm_runs.csv"),
        "LSTM V2 (exp_lstm_5seeds)",
    ),
    'gru': (
        os.path.join(MODELOS_DIR, "GRU", "exp_gru_5seeds", "gru_runs.csv"),
        "GRU V3 (exp_gru_5seeds)",
    ),
}

METRICS = ['accuracy', 'f1_macro', 'f1_weighted']
METRIC_LABELS = {'accuracy': 'Accuracy', 'f1_macro': 'F1 macro',
                  'f1_weighted': 'F1 weighted'}


# ── Utilitários ───────────────────────────────────────────────────────────────

def load_runs(csv_path, label):
    if not os.path.exists(csv_path):
        print(f"[pulando] {label}: arquivo nao encontrado ({csv_path})")
        return None
    df = pd.read_csv(csv_path).sort_values('seed').reset_index(drop=True)
    print(f"[OK] {label}: {len(df)} execucoes  ({csv_path})")
    return df


def wilcoxon_safe(a, b):
    """Wilcoxon signed-rank two-sided; retorna (stat, p). Trata empate perfeito."""
    if np.all(np.array(a) == np.array(b)):
        return float('nan'), 1.0
    try:
        return wilcoxon(a, b, alternative='two-sided')
    except Exception:
        return float('nan'), float('nan')


def interpret_p(p):
    if p != p:   return "n/a"
    if p < 0.01: return "p<0.01 ✓✓"
    if p < 0.05: return "p<0.05 ✓"
    if p < 0.10: return "p<0.10 ~"
    return f"p={p:.3f} ✗"


# ── Comparação de um par (baseline vs V5) ───────────────────────────────────

def compare_pair(baseline_df, v5_df, baseline_label, v5_label):
    print()
    print("=" * 76)
    print(f"{baseline_label}  vs  PatchTST V5 ({v5_label}) — Wilcoxon signed-rank")
    print("=" * 76)

    n = min(len(baseline_df), len(v5_df))
    if len(baseline_df) != len(v5_df):
        print(f"AVISO: N diferente ({len(baseline_df)} vs {len(v5_df)}) — usando primeiros {n} pares.")
    b_df, v_df = baseline_df.iloc[:n], v5_df.iloc[:n]

    print(f"\n{'Metrica':<14}  {'Baseline (mu +/- sigma)':>24}  {'V5 (mu +/- sigma)':>20}  "
          f"{'Delta':>8}  Wilcoxon")
    print("-" * 76)

    rows = []
    for m in METRICS:
        if m not in b_df.columns or m not in v_df.columns:
            print(f"  {m}: coluna ausente — pulando")
            continue

        b_v, v_v   = b_df[m].values, v_df[m].values
        mu_b, sg_b = b_v.mean(), b_v.std(ddof=1)
        mu_v, sg_v = v_v.mean(), v_v.std(ddof=1)
        delta      = mu_v - mu_b
        stat, p    = wilcoxon_safe(v_v, b_v)

        print(f"  {METRIC_LABELS.get(m, m):<12}  "
              f"{mu_b:+.4f} +/- {sg_b:.4f}         "
              f"{mu_v:+.4f} +/- {sg_v:.4f}  "
              f"{delta:+.6f}  {interpret_p(p)}")

        rows.append({'baseline': baseline_label, 'metric': m,
                      'baseline_mean': mu_b, 'baseline_std': sg_b,
                      'v5_mean': mu_v, 'v5_std': sg_v, 'delta': delta,
                      'wilcoxon_stat': stat, 'p_value': p})

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v5-exp', default='exp_warmup',
                         help="Subpasta do experimento V5 em ModelosNew/Transformer/ "
                              "(default: melhor config, seq_len=16 + OneCycleLR)")
    parser.add_argument('--baselines', default='tcn,lstm,gru',
                         help="Lista separada por virgula entre: tcn,lstm,gru")
    args = parser.parse_args()

    v5_df = load_runs(
        os.path.join(TRANSFORMER_DIR, args.v5_exp, "patchtst_v5_runs.csv"),
        f"PatchTST V5 ({args.v5_exp})",
    )
    if v5_df is None:
        sys.exit(f"\nExperimento V5 '{args.v5_exp}' nao encontrado. "
                  f"Rode o treino do Transformer primeiro.")

    requested = [b.strip() for b in args.baselines.split(',') if b.strip()]
    all_results = []

    for name in requested:
        if name not in BASELINES:
            print(f"[!] baseline desconhecida: '{name}' — ignorando "
                  f"(opcoes: {', '.join(BASELINES)})")
            continue
        csv_path, label = BASELINES[name]
        b_df = load_runs(csv_path, label)
        if b_df is None:
            print(f"    -> rode 'python scripts/run_{name}_5seeds.py' para gerar esse CSV")
            continue

        result_df = compare_pair(b_df, v5_df, label, args.v5_exp)
        all_results.append(result_df)

        out_dir = os.path.dirname(csv_path)
        out_csv = os.path.join(out_dir, f"wilcoxon_{name}_vs_{args.v5_exp}.csv")
        result_df.to_csv(out_csv, index=False)
        print(f"\nResultado salvo: {out_csv}")

    if not all_results:
        sys.exit("\nNenhuma baseline com dados disponiveis. Nada a comparar.")

    print("\n" + "=" * 76)
    print(f"RESUMO — todas as baselines vs PatchTST V5 ({args.v5_exp})")
    print("=" * 76)
    combined = pd.concat(all_results, ignore_index=True)
    pivot = combined.pivot(index='baseline', columns='metric', values='delta')
    pivot = pivot[[m for m in METRICS if m in pivot.columns]]
    print(f"\nDelta (V5 - baseline), positivo = V5 melhor:\n")
    print(pivot.to_string(float_format=lambda x: f"{x:+.4f}"))

    combined_csv = os.path.join(TRANSFORMER_DIR, f"wilcoxon_all_baselines_vs_{args.v5_exp}.csv")
    combined.to_csv(combined_csv, index=False)
    print(f"\nTabela combinada salva: {combined_csv}")
    print("\nLegenda: ✓✓ p<0.01  ✓ p<0.05  ~ tendencia (p<0.10)  ✗ nao significativo")


if __name__ == "__main__":
    main()
