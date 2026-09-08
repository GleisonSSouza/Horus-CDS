#!/usr/bin/env python3
"""LSTM (V2) com protocolo de 5 seeds — mesma metodologia do TCN corrigido.

Espelha `run_tcn_5seeds_fixed.py` linha a linha na parte de protocolo
experimental (split, features, class weights, thresholds fit-no-treino,
5 seeds, agregação μ±σ). A única coisa que muda é a arquitetura do modelo:
aqui é o LSTM de regressão de `ThreadTrain/LSTM_corrigido.py`, sem a busca
de hiperparâmetros (`RandomizedSearchCV`) do arquivo original — hiperparâmetros
fixos para manter apples-to-apples entre seeds e entre modelos (TCN/LSTM/GRU
usam o mesmo `TRAIN_CONFIG`).

Por que thresholds fit-no-treino (não no teste): ver o header de
`run_tcn_5seeds_fixed.py` — calcular percentis sobre as predições do teste
é uma forma de vazamento e apaga a variação genuína entre seeds (rank order
de um regressor bem treinado é quase invariante à inicialização). Aqui os
thresholds são ajustados nas predições de TREINO e aplicados congelados
no teste, restaurando desvio-padrão > 0 entre seeds.

Uso:
    cd Horus-CDS && source .venv/bin/activate
    python scripts/run_lstm_5seeds.py

Saída: root/Linux/DadosDoPostreino/ModelosNew/LSTM/exp_lstm_5seeds/
    lstm_runs.csv, lstm_summary.csv, lstm_seed{N}_report.txt
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
from tensorflow.keras.models import Sequential          # type: ignore
from tensorflow.keras import layers, optimizers         # type: ignore
from tensorflow.keras.callbacks import LambdaCallback    # type: ignore


# ── Configuração ──────────────────────────────────────────────────────────────

CSV_PATH   = os.path.join(LINUX_DIR, "DadosReais", "dados_normalizados_smartgrid.csv")
OUTPUT_DIR = os.path.join(LINUX_DIR, "DadosDoPostreino", "ModelosNew", "LSTM", "exp_lstm_5seeds")
SEEDS      = [0, 1, 2, 3, 4]

# Arquitetura de ThreadTrain/LSTM_corrigido.py (KerasLSTMRegressorBalanced),
# hiperparâmetros fixados nos valores default usados pelo RandomizedSearchCV
# original (units=50, tanh, dropout=0.2, noise_level=0.1 — o run() do arquivo
# legado chama com noise_level=0.1 mesmo o default da classe sendo 0.5).
LSTM_CONFIG = dict(units=50, activation='tanh', dropout=0.2, noise_level=0.1)

# Idêntico ao TCN_CONFIG/TRAIN_CONFIG de run_tcn_5seeds_fixed.py — garante
# que TCN, LSTM e GRU sejam comparáveis sob o mesmo orçamento de treino.
TRAIN_CONFIG = dict(
    n_epochs=50, batch_size=64, lr=0.001, patience=5, val_ratio=0.15,
)

# Percentis para converter saída de regressão em classes (mesmos do TCN e
# do LSTM/GRU legados, calibrados para reproduzir a proporção de classes).
PERCENTILE_ILEGAL   = 17.8
PERCENTILE_SUSPEITO = 23.2


# ── Utilitários ───────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def build_lstm_model(input_shape, config, lr):
    """Réplica da KerasLSTMRegressorBalanced de LSTM_corrigido.py, sem sklearn wrapper.

    Diferença deliberada vs. o original: `recurrent_dropout=0.0` (o legado usava
    `recurrent_dropout=config['dropout']`). Com `recurrent_dropout > 0` o Keras
    desativa o fused kernel e cai numa implementação genérica célula-a-célula —
    no backend `tensorflow-metal` (Mac Apple Silicon) isso é ~20-50x mais lento,
    a ponto de um epoch inteiro parecer travado (ver relato do Gleison, 2026-07-29:
    5 min sem nenhum print e sem terminar a seed 0). `dropout` (nas conexões de
    entrada, não recorrentes) continua ativo e não tem esse problema de performance.
    """
    model = Sequential([
        layers.GaussianNoise(config['noise_level'], input_shape=input_shape),

        layers.LSTM(config['units'], activation=config['activation'],
                    dropout=config['dropout'], recurrent_dropout=0.0,
                    return_sequences=True),
        layers.BatchNormalization(),

        layers.LSTM(config['units'] // 2, activation=config['activation'],
                    dropout=config['dropout'], recurrent_dropout=0.0,
                    return_sequences=False),
        layers.BatchNormalization(),
        layers.Dropout(config['dropout']),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(config['dropout']),

        layers.Dense(32, activation='relu'),
        layers.Dropout(config['dropout']),

        layers.Dense(1),
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='mean_squared_error', metrics=['mse'])
    return model


def fit_thresholds(y_pred_train_flat, p_ilegal=PERCENTILE_ILEGAL,
                    p_suspeito=PERCENTILE_SUSPEITO):
    """Calcula thresholds SOMENTE a partir das predições do TREINO ("fit")."""
    t1 = np.percentile(y_pred_train_flat, p_ilegal)
    t2 = np.percentile(y_pred_train_flat, p_suspeito)
    return t1, t2


def apply_thresholds(y_pred_flat, t1, t2):
    """Classifica predições usando thresholds já congelados ("transform")."""
    return np.where(y_pred_flat < t1, 'ilegal',
                    np.where(y_pred_flat < t2, 'suspeito', 'válido'))


# ── Treino de uma execução ─────────────────────────────────────────────────────

def train_one_run(seed, X_train, y_train, X_val, y_val, X_test,
                   categories_test, class_weights, model_config, train_config):
    """Treina o LSTM para uma seed e retorna métricas e classification report."""
    set_seed(seed)

    def approx_class(val):
        if val < -0.5: return 'ilegal'
        if val <  0.5: return 'suspeito'
        return 'válido'

    sample_weights = np.array([
        class_weights.get(approx_class(v), 1.0) for v in y_train.flatten()
    ])

    model      = build_lstm_model((X_train.shape[1], 1), model_config, train_config['lr'])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=train_config['patience'],
        restore_best_weights=True, verbose=0,
    )
    # Heartbeat: sem isso, model.fit(verbose=0) fica em silencio total ate
    # terminar (ou dar early stopping), o que pareceu "travado" na pratica.
    heartbeat = LambdaCallback(on_epoch_end=lambda epoch, logs: print(
        f"    [seed={seed}] epoch {epoch + 1}: "
        f"loss={logs.get('loss'):.4f}  val_loss={logs.get('val_loss'):.4f}",
        flush=True,
    ))

    t_start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=train_config['n_epochs'], batch_size=train_config['batch_size'],
        validation_data=(X_val, y_val), sample_weight=sample_weights,
        callbacks=[early_stop, heartbeat], verbose=0,
    )
    train_elapsed    = time.time() - t_start
    n_epochs_trained = len(history.history['loss'])

    y_pred_train_raw = model.predict(X_train, verbose=0).flatten()
    t1, t2           = fit_thresholds(y_pred_train_raw)

    y_pred_test_raw = model.predict(X_test, verbose=0).flatten()
    y_pred_classes  = apply_thresholds(y_pred_test_raw, t1, t2)

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
    print("LSTM V2 — protocolo de 5 seeds (thresholds fit no treino)")
    print("=" * 70)

    print("\n[1/4] Pre-processamento")
    train_df, test_df = temporal_split(data, model_name="LSTM-5seeds")
    class_weights      = analyze_class_distribution(train_df, model_name="LSTM-5seeds")
    train_df = add_temporal_features_safe(train_df, is_train=True,  model_name="LSTM-5seeds")
    test_df  = add_temporal_features_safe(test_df,  is_train=False, model_name="LSTM-5seeds")

    split_idx  = int(len(train_df) * (1 - TRAIN_CONFIG['val_ratio']))
    train_part = train_df.iloc[:split_idx].copy()
    val_part   = train_df.iloc[split_idx:].copy()
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
            model_config=LSTM_CONFIG, train_config=TRAIN_CONFIG,
        )
        runs[seed] = metrics

        with open(os.path.join(OUTPUT_DIR, f"lstm_seed{seed}_report.txt"), 'w') as f:
            f.write(f"LSTM V2 (thresholds fit no treino) — seed={seed}\n{'='*50}\n")
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
    runs_df.to_csv(os.path.join(OUTPUT_DIR, "lstm_runs.csv"), index=False)

    summary_df = summarize_runs(runs_df)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "lstm_summary.csv"), index=False)

    for _, row in summary_df.iterrows():
        print(f"  {row['metric']:14s}: {row['mean']:.4f} +/- {row['std']:.4f}"
              f"  (min={row['min']:.4f}, max={row['max']:.4f})")

    print("\n" + "=" * 70)
    print("Proximo passo: python scripts/run_wilcoxon_baselines.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
