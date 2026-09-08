#!/usr/bin/env python3
"""TCN V4 com protocolo de 5 seeds — versão CORRIGIDA do threshold.

Este script é uma cópia de `run_tcn_5seeds.py` com uma única mudança
metodológica: os thresholds de percentil que convertem a saída de regressão
em classes deixam de ser calculados sobre as predições do CONJUNTO DE TESTE
e passam a ser calculados sobre as predições do CONJUNTO DE TREINO, depois
aplicados de forma congelada no teste.

Por que isso importa (ver docs/Relatorio_Andamento_TCC_2026-07-02.docx, Seção 6):

`run_tcn_5seeds.py` calcula `t1 = percentile(y_pred_TESTE, 17.8)` — uma
operação invariante à ordem relativa (rank order) dos valores preditos.
Um regressor bem treinado preserva o rank order do test set mesmo com pesos
de inicialização diferentes entre seeds, então a classificação final acaba
sendo IDÊNTICA nas 5 execuções (desvio-padrão zero), apesar do treino em si
ser genuinamente estocástico (tempo de treino e epochs variam normalmente).
Além disso, calcular o threshold sobre o teste é uma forma de vazamento:
a fronteira de decisão é ajustada olhando para os dados que serão avaliados.

A correção aqui aplicada segue o mesmo princípio já usado pelo StandardScaler
no restante do script (fit só no treino, transform no teste):

    t1, t2 = percentile(y_pred_TREINO, 17.8), percentile(y_pred_TREINO, 23.2)
    y_pred_classes_teste = classifica(y_pred_TESTE, t1, t2)   # t1/t2 fixos

Isso remove o vazamento E restaura a variação genuína entre seeds, porque
cada seed treina pesos diferentes → gera uma distribuição de predições no
treino diferente → produz t1/t2 diferentes → o MESMO conjunto de teste pode
ser classificado de forma diferente por seed.

Resultado é salvo em pasta separada (`exp_tcn_5seeds_fixed/`) para preservar
o experimento original (`exp_tcn_5seeds/`) como evidência do achado.

Uso:
    cd Horus-CDS && source .venv/bin/activate
    python scripts/run_tcn_5seeds_fixed.py
"""

import os, sys, random, time
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINUX_DIR    = os.path.join(PROJECT_ROOT, "root", "Linux")
sys.path.insert(0, LINUX_DIR)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

from TratamentoDeDados.temporal_split import (
    temporal_split, add_temporal_features_safe,
    analyze_class_distribution, DEFAULT_FEATURES, DEFAULT_TARGET,
)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential   # type: ignore
from tensorflow.keras import layers, optimizers  # type: ignore
from tcn import TCN                              # type: ignore


# ── Configuração ──────────────────────────────────────────────────────────────

CSV_PATH   = os.path.join(LINUX_DIR, "DadosReais", "dados_normalizados_smartgrid.csv")
OUTPUT_DIR = os.path.join(LINUX_DIR, "DadosDoPostreino", "ModelosNew", "TCN", "exp_tcn_5seeds_fixed")
SEEDS      = [0, 1, 2, 3, 4]

TCN_CONFIG = dict(
    nb_filters=64, kernel_size=3, nb_stacks=1,
    dilations=[1, 2, 4, 8], activation='relu', use_skip_connections=True,
)
TRAIN_CONFIG = dict(
    n_epochs=50, batch_size=64, lr=0.001, patience=5, val_ratio=0.15,
)

# Percentis para converter saída de regressão em classes (mesmos do original,
# calibrados para reproduzir a proporção de classes do treino).
PERCENTILE_ILEGAL   = 17.8
PERCENTILE_SUSPEITO = 23.2


# ── Utilitários ───────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def build_tcn_model(input_shape, config, lr):
    model = Sequential([
        TCN(input_shape=input_shape, nb_filters=config['nb_filters'],
            kernel_size=config['kernel_size'], nb_stacks=config['nb_stacks'],
            dilations=config['dilations'], activation=config['activation'],
            use_skip_connections=config['use_skip_connections']),
        layers.Dense(1),
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='mean_squared_error', metrics=['mse'])
    return model


def fit_thresholds(y_pred_train_flat, p_ilegal=PERCENTILE_ILEGAL,
                    p_suspeito=PERCENTILE_SUSPEITO):
    """Calcula os thresholds SOMENTE a partir das predições do TREINO.

    Esta é a metade "fit" — nunca deve receber predições do teste.
    """
    t1 = np.percentile(y_pred_train_flat, p_ilegal)
    t2 = np.percentile(y_pred_train_flat, p_suspeito)
    return t1, t2


def apply_thresholds(y_pred_flat, t1, t2):
    """Classifica predições usando thresholds JÁ CONGELADOS (vindos do treino).

    Esta é a metade "transform" — pode receber predições do treino ou do
    teste; os thresholds em si não são recalculados aqui.
    """
    return np.where(y_pred_flat < t1, 'ilegal',
                    np.where(y_pred_flat < t2, 'suspeito', 'válido'))


# ── Treino de uma execução ─────────────────────────────────────────────────────

def train_one_run(seed, X_train, y_train, X_val, y_val, X_test,
                  categories_test, class_weights, tcn_config, train_config):
    """Treina o TCN para uma seed e retorna métricas e classification report.

    Args:
        seed (int): seed para reproducibilidade.
        X_train, y_train: arrays de treino.
        X_val, y_val: arrays de validação para early stopping.
        X_test: array de teste.
        categories_test (np.ndarray): rótulos reais do teste (strings).
        class_weights (dict): pesos por classe.
        tcn_config (dict): hiperparâmetros da arquitetura TCN.
        train_config (dict): lr, n_epochs, batch_size, patience.

    Returns:
        tuple: (metrics dict, classification_report str, y_pred ndarray)
    """
    set_seed(seed)

    # Aproxima classe a partir de y contínuo para aplicar class_weights como sample_weights
    def approx_class(val):
        if val < -0.5: return 'ilegal'
        if val <  0.5: return 'suspeito'
        return 'válido'

    sample_weights = np.array([
        class_weights.get(approx_class(v), 1.0) for v in y_train.flatten()
    ])

    model      = build_tcn_model((X_train.shape[1], 1), tcn_config, train_config['lr'])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=train_config['patience'],
        restore_best_weights=True, verbose=0,
    )

    t_start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=train_config['n_epochs'], batch_size=train_config['batch_size'],
        validation_data=(X_val, y_val), sample_weight=sample_weights,
        callbacks=[early_stop], verbose=0,
    )
    train_elapsed    = time.time() - t_start
    n_epochs_trained = len(history.history['loss'])

    # --- FIX: thresholds vêm do TREINO, nunca do teste ---
    y_pred_train_raw = model.predict(X_train, verbose=0).flatten()
    t1, t2           = fit_thresholds(y_pred_train_raw)

    y_pred_test_raw = model.predict(X_test, verbose=0).flatten()
    y_pred_classes  = apply_thresholds(y_pred_test_raw, t1, t2)
    # --- fim do fix (resto idêntico ao script original) ---

    labels = ['ilegal', 'suspeito', 'válido']
    acc    = accuracy_score(categories_test, y_pred_classes)
    f1m    = f1_score(categories_test, y_pred_classes, labels=labels,
                      average='macro', zero_division=0)
    f1w    = f1_score(categories_test, y_pred_classes, labels=labels,
                      average='weighted', zero_division=0)
    report = classification_report(categories_test, y_pred_classes,
                                   labels=labels, zero_division=0)

    metrics = {
        'seed': seed, 'accuracy': acc, 'f1_macro': f1m, 'f1_weighted': f1w,
        'threshold_ilegal': t1, 'threshold_suspeito': t2,
        'n_epochs_trained': n_epochs_trained, 'train_time_s': train_elapsed,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
    }

    return metrics, report, y_pred_classes


# ── Agregação ─────────────────────────────────────────────────────────────────

def summarize_runs(runs_df):
    """Retorna DataFrame com média ± desvio (ddof=1) por métrica."""
    metric_cols = ['accuracy', 'f1_macro', 'f1_weighted', 'train_time_s']
    rows = []
    for m in metric_cols:
        vals = runs_df[m].dropna().values
        rows.append({
            'metric': m,
            'mean':   float(np.mean(vals)),
            'std':    float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'min':    float(np.min(vals)), 'max': float(np.max(vals)),
            'n_runs': int(len(vals)),
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data = pd.read_csv(CSV_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("TCN V4 — protocolo de 5 seeds (threshold CORRIGIDO: fit no treino)")
    print("=" * 70)

    print("\n[1/4] Pre-processamento")
    train_df, test_df = temporal_split(data, model_name="TCN-5seeds-fixed")
    class_weights     = analyze_class_distribution(train_df, model_name="TCN-5seeds-fixed")
    train_df = add_temporal_features_safe(train_df, is_train=True,  model_name="TCN-5seeds-fixed")
    test_df  = add_temporal_features_safe(test_df,  is_train=False, model_name="TCN-5seeds-fixed")

    split_idx   = int(len(train_df) * (1 - TRAIN_CONFIG['val_ratio']))
    train_part  = train_df.iloc[:split_idx].copy()
    val_part    = train_df.iloc[split_idx:].copy()
    print(f"Treino: {len(train_part)}  Val: {len(val_part)}  Teste: {len(test_df)}")

    print("\n[2/4] Features e normalizacao")
    scaler = Pipeline([('imputer', SimpleImputer(strategy='mean')),
                       ('scaler', StandardScaler())])
    X_train = np.expand_dims(scaler.fit_transform(train_part[DEFAULT_FEATURES].values), -1)
    X_val   = np.expand_dims(scaler.transform(val_part[DEFAULT_FEATURES].values), -1)
    X_test  = np.expand_dims(scaler.transform(test_df[DEFAULT_FEATURES].values), -1)
    y_train = train_part[DEFAULT_TARGET].values.reshape(-1, 1)
    y_val   = val_part[DEFAULT_TARGET].values.reshape(-1, 1)
    categories_test = test_df['CATEGORY'].values
    print(f"X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    print(f"\n[3/4] Loop de {len(SEEDS)} execucoes")
    runs = {}

    for i, seed in enumerate(SEEDS):
        print(f"\n--- Execucao {i+1}/{len(SEEDS)} (seed={seed}) ---")
        metrics, report, _ = train_one_run(
            seed=seed,
            X_train=X_train, y_train=y_train,
            X_val=X_val,     y_val=y_val,
            X_test=X_test,   categories_test=categories_test,
            class_weights=class_weights,
            tcn_config=TCN_CONFIG, train_config=TRAIN_CONFIG,
        )
        runs[seed] = metrics

        with open(os.path.join(OUTPUT_DIR, f"tcn_seed{seed}_report.txt"), 'w') as f:
            f.write(f"TCN V4 (threshold corrigido) — seed={seed}\n{'='*50}\n")
            for k in ('accuracy', 'f1_macro', 'f1_weighted',
                      'n_epochs_trained', 'threshold_ilegal', 'threshold_suspeito'):
                f.write(f"{k} = {metrics[k]:.4f}\n")
            f.write(f"\n{report}")

        print(f"[seed={seed}] acc={metrics['accuracy']:.4f}  "
              f"f1_macro={metrics['f1_macro']:.4f}  "
              f"t1={metrics['threshold_ilegal']:.4f}  "
              f"t2={metrics['threshold_suspeito']:.4f}  "
              f"epochs={metrics['n_epochs_trained']}  "
              f"t={metrics['train_time_s']:.1f}s")

    print("\n[4/4] Agregacao")
    runs_df = pd.DataFrame(runs.values())
    runs_df.to_csv(os.path.join(OUTPUT_DIR, "tcn_runs.csv"), index=False)

    summary_df = summarize_runs(runs_df)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "tcn_summary.csv"), index=False)

    for _, row in summary_df.iterrows():
        print(f"  {row['metric']:14s}: {row['mean']:.4f} +/- {row['std']:.4f}"
              f"  (min={row['min']:.4f}, max={row['max']:.4f})")

    acc_std = float(summary_df.loc[summary_df['metric'] == 'accuracy', 'std'].iloc[0])
    print("\n" + "=" * 70)
    if acc_std == 0.0:
        print("[!] AVISO: desvio-padrao ainda ZERO em accuracy. O fix nao teve o efeito")
        print("    esperado — investigar antes de seguir (ver Secao 6 do relatorio).")
    else:
        print(f"[OK] Desvio-padrao de accuracy agora e {acc_std:.4f} (> 0) — a")
        print("    estocasticidade do treino deixou de ser apagada pelo threshold.")
    print("=" * 70)

    print("\nProximo passo: python scripts/run_wilcoxon.py --tcn-exp exp_tcn_5seeds_fixed --v5-exp exp_warmup")
    print("=" * 70)


if __name__ == "__main__":
    main()
