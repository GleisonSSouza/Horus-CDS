#!/usr/bin/env python3
"""Avalia o desempenho POR CLASSE (ilegal / suspeito / válido) dos checkpoints
já treinados do PatchTST V5, para comparar se o warmup (OneCycleLR) realmente
melhorou a classe "suspeito" -- métrica que não é salva nos summary.csv
(esses só têm accuracy/f1_macro/f1_weighted/auc agregados, sem quebra por classe).

Reconstrói o mesmo conjunto de teste usado no treino (pré-processamento
determinístico, não depende de seed) e roda os 5 checkpoints salvos de cada
experimento, calculando precision/recall/f1 por classe.

Uso:
    cd Horus-CDS && python scripts/eval_per_class.py exp_seqlen16 exp_warmup
    cd Horus-CDS && python scripts/eval_per_class.py  # roda os defaults abaixo
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINUX_DIR    = os.path.join(PROJECT_ROOT, "root", "Linux")
sys.path.insert(0, LINUX_DIR)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

from TratamentoDeDados.temporal_split import (
    temporal_split,
    add_temporal_features_safe,
)
from TratamentoDeDados.windowing import (
    make_windows,
    INT_TO_CATEGORY,
    DEFAULT_TRANSFORMER_FEATURES,
)
from ThreadTrain.Transformer_PatchTST import PatchTST

CSV_PATH  = os.path.join(LINUX_DIR, "DadosReais", "dados_normalizados_smartgrid.csv")
BASE_DIR  = os.path.join(LINUX_DIR, "DadosDoPostreino", "ModelosNew", "Transformer")
SEEDS     = [0, 1, 2, 3, 4]
LABELS    = [INT_TO_CATEGORY[i] for i in range(3)]  # ['ilegal', 'suspeito', 'válido']

# Experimentos a comparar por default (mesma feature_cols/seq_len=16,
# única diferença é o scheduler ligado ou não).
DEFAULT_EXPS = ["exp_seqlen16", "exp_warmup"]


def get_test_set(seq_len, feature_cols):
    data = pd.read_csv(CSV_PATH)
    train_df, test_df = temporal_split(data, model_name="eval_per_class")
    test_df = add_temporal_features_safe(test_df, is_train=False, model_name="eval_per_class")
    X_test, y_test = make_windows(test_df, feature_cols=feature_cols, seq_len=seq_len,
                                   model_name="test")
    return X_test, y_test


def eval_experiment(exp_name, feature_cols=None):
    exp_dir = os.path.join(BASE_DIR, exp_name)
    if not os.path.isdir(exp_dir):
        print(f"[!] pasta nao encontrada: {exp_dir}")
        return None

    feature_cols = feature_cols or list(DEFAULT_TRANSFORMER_FEATURES)
    device = torch.device("cpu")

    per_class_rows = []
    cached_test = {}  # seq_len -> (X_test, y_test)

    for seed in SEEDS:
        ckpt_path = os.path.join(exp_dir, f"patchtst_v5_seed{seed}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[!] checkpoint ausente: {ckpt_path}")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model_config = ckpt["model_config"]
        seq_len = model_config["seq_len"]

        if seq_len not in cached_test:
            cached_test[seq_len] = get_test_set(seq_len, feature_cols)
        X_test, y_test = cached_test[seq_len]

        model = PatchTST(**model_config).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        with torch.no_grad():
            logits = model(torch.tensor(X_test, dtype=torch.float32))
            preds  = logits.argmax(dim=-1).numpy()

        report = classification_report(
            y_test, preds, labels=[0, 1, 2], target_names=LABELS,
            output_dict=True, zero_division=0,
        )
        for label in LABELS:
            per_class_rows.append({
                "experiment": exp_name, "seed": seed, "classe": label,
                "precision": report[label]["precision"],
                "recall":    report[label]["recall"],
                "f1":        report[label]["f1-score"],
                "support":   report[label]["support"],
            })
        print(f"[{exp_name}] seed={seed}  suspeito: "
              f"P={report['suspeito']['precision']:.3f}  "
              f"R={report['suspeito']['recall']:.3f}  "
              f"F1={report['suspeito']['f1-score']:.3f}")

    return pd.DataFrame(per_class_rows)


def main():
    exp_names = sys.argv[1:] or DEFAULT_EXPS
    all_rows = []
    for exp_name in exp_names:
        df = eval_experiment(exp_name)
        if df is not None:
            all_rows.append(df)

    if not all_rows:
        print("Nenhum experimento avaliado.")
        return

    full_df = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(BASE_DIR, "per_class_comparison.csv")
    full_df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print("Resumo por classe (media +/- desvio entre seeds)")
    print("=" * 70)
    summary = (full_df.groupby(["experiment", "classe"])[["precision", "recall", "f1"]]
               .agg(["mean", "std"]))
    print(summary)
    print(f"\nSalvo em: {out_path}")


if __name__ == "__main__":
    main()
