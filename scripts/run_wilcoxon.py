#!/usr/bin/env python3
"""Comparação estatística TCN V4 vs PatchTST V5 — Wilcoxon signed-rank.

Lê os CSVs de runs e imprime tabela comparativa com p-values.

Por padrão compara o TCN com threshold CORRIGIDO (`exp_tcn_5seeds_fixed`,
ver `run_tcn_5seeds_fixed.py`) contra a melhor config do V5 (`exp_warmup`).
Use `--tcn-exp exp_tcn_5seeds` para reproduzir a comparação antiga (inválida,
threshold calculado no teste — mantido só para referência histórica).

Uso:
    cd Horus-CDS && python scripts/run_wilcoxon.py
    cd Horus-CDS && python scripts/run_wilcoxon.py --tcn-exp exp_tcn_5seeds --v5-exp exp_optimal
"""

import argparse, os, sys

PROJECT_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINUX_DIR       = os.path.join(PROJECT_ROOT, "root", "Linux")
TRANSFORMER_DIR = os.path.join(LINUX_DIR, "DadosDoPostreino", "ModelosNew", "Transformer")
TCN_BASE_DIR    = os.path.join(LINUX_DIR, "DadosDoPostreino", "ModelosNew", "TCN")

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ── Constantes ────────────────────────────────────────────────────────────────

METRICS = ['accuracy', 'f1_macro', 'f1_weighted']
METRIC_LABELS = {'accuracy': 'Accuracy', 'f1_macro': 'F1 macro',
                 'f1_weighted': 'F1 weighted'}


# ── Utilitários ───────────────────────────────────────────────────────────────

def load_runs(csv_path, label):
    if not os.path.exists(csv_path):
        sys.exit(f"Arquivo não encontrado: {csv_path}\n"
                 f"Execute o script de treino correspondente primeiro.")
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


# ── Comparação ────────────────────────────────────────────────────────────────

def compare(tcn_df, v5_df, v5_exp_name):
    print()
    print("=" * 72)
    print(f"TCN V4 vs PatchTST V5 ({v5_exp_name}) — Wilcoxon signed-rank, N={len(tcn_df)}")
    print("=" * 72)
    print(f"\n{'Metrica':<14}  {'TCN (mu +/- sigma)':>20}  {'V5 (mu +/- sigma)':>20}  "
          f"{'Delta':>8}  Wilcoxon")
    print("-" * 72)

    results = []
    for m in METRICS:
        if m not in tcn_df.columns or m not in v5_df.columns:
            print(f"  {m}: coluna ausente — pulando")
            continue

        tcn_v, v5_v = tcn_df[m].values, v5_df[m].values
        mu_t, sg_t  = tcn_v.mean(), tcn_v.std(ddof=1)
        mu_v, sg_v  = v5_v.mean(),  v5_v.std(ddof=1)
        delta       = mu_v - mu_t
        stat, p     = wilcoxon_safe(v5_v, tcn_v)

        print(f"  {METRIC_LABELS.get(m, m):<12}  "
              f"{mu_t:+.4f} +/- {sg_t:.4f}  "
              f"{mu_v:+.4f} +/- {sg_v:.4f}  "
              f"{delta:+.6f}  {interpret_p(p)}")

        results.append({'metric': m, 'tcn_mean': mu_t, 'tcn_std': sg_t,
                        'v5_mean': mu_v, 'v5_std': sg_v, 'delta': delta,
                        'wilcoxon_stat': stat, 'p_value': p})

    print("\nLegenda: ✓✓ p<0.01  ✓ p<0.05  ~ tendencia (p<0.10)  ✗ nao significativo")

    print(f"\n{'seed':<6}  {'TCN acc':>8}  {'V5 acc':>8}  "
          f"{'TCN f1m':>8}  {'V5 f1m':>8}")
    print("-" * 44)
    for i in range(len(tcn_df)):
        s = int(tcn_df.iloc[i]['seed'])
        print(f"{s:<6}  {tcn_df.iloc[i]['accuracy']:>8.4f}  "
              f"{v5_df.iloc[i]['accuracy']:>8.4f}  "
              f"{tcn_df.iloc[i]['f1_macro']:>8.4f}  "
              f"{v5_df.iloc[i]['f1_macro']:>8.4f}")

    return pd.DataFrame(results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tcn-exp', default='exp_tcn_5seeds_fixed',
                        help="Subpasta do experimento TCN em ModelosNew/TCN/ "
                             "(default: versao com threshold corrigido)")
    parser.add_argument('--v5-exp', default='exp_warmup',
                        help="Subpasta do experimento V5 em ModelosNew/Transformer/ "
                             "(default: melhor config, seq_len=16 + OneCycleLR)")
    args = parser.parse_args()

    tcn_dir = os.path.join(TCN_BASE_DIR, args.tcn_exp)
    tcn_df  = load_runs(os.path.join(tcn_dir, "tcn_runs.csv"), f"TCN V4 ({args.tcn_exp})")
    v5_df   = load_runs(
        os.path.join(TRANSFORMER_DIR, args.v5_exp, "patchtst_v5_runs.csv"),
        f"PatchTST V5 ({args.v5_exp})",
    )

    if args.tcn_exp == 'exp_tcn_5seeds' and tcn_df['accuracy'].std(ddof=1) == 0.0:
        print("\n[!] AVISO: essa pasta contem a versao com o bug de threshold conhecido")
        print("    (desvio-padrao zero — ver Secao 6 do relatorio de andamento).")
        print("    Wilcoxon nao e estatisticamente valido aqui. Use --tcn-exp")
        print("    exp_tcn_5seeds_fixed apos rodar run_tcn_5seeds_fixed.py.\n")

    if len(tcn_df) != len(v5_df):
        n      = min(len(tcn_df), len(v5_df))
        tcn_df = tcn_df.iloc[:n]
        v5_df  = v5_df.iloc[:n]
        print(f"AVISO: N diferente — usando primeiros {n} pares.")

    results_df = compare(tcn_df, v5_df, args.v5_exp)
    out_csv    = os.path.join(tcn_dir, f"wilcoxon_tcn_vs_{args.v5_exp}.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nResultado salvo: {out_csv}")


if __name__ == "__main__":
    main()
