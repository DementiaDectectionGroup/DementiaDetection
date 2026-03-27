import numpy as np
from sklearn.preprocessing import StandardScaler
"""
DementiaPred – Phase 1 main entry point
========================================
Runs stratified 5-fold CV for all models, reports mean ± std for
Accuracy / F1 / ROC-AUC, and saves per-fold results to CSV.

Usage:
    python main.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dataprocess import load_or_extract_features, get_kfold_splits
from modelfactory import ModelFactory

# ── Configuration ────────────────────────────────────────────────────────────
TRAIN_AUDIO_DIR = "ADReSSo/train/audio"
CACHE_FILE      = "train_features_cache.npz"
N_SPLITS        = 5
OUTPUT_CSV      = "cv_results.csv"

MODELS_TO_TEST  = ModelFactory.all_model_names()   # run every registered model


# ── Helpers ──────────────────────────────────────────────────────────────────
def evaluate_pipeline(pipeline, X_train, y_train, X_val, y_val):
    """Fit pipeline and return accuracy, f1, roc_auc."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1  = f1_score(y_val, y_pred, zero_division=0)

    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        auc    = roc_auc_score(y_val, y_prob)
    else:
        auc = float('nan')

    return acc, f1, auc


def print_summary(results: dict):
    print("\n" + "=" * 55)
    print(f"{'Final Cross-Validation Results':^55}")
    print("=" * 55)
    fmt = "{:<22}  {:>7}  {:>7}  {:>7}"
    print(fmt.format("Model", "Acc", "F1", "AUC"))
    print("-" * 55)
    for model_name, folds in results.items():
        acc = np.array(folds['accuracy'])
        f1  = np.array(folds['f1_score'])
        auc = np.array(folds['roc_auc'])
        print(fmt.format(
            model_name,
            f"{acc.mean():.3f}±{acc.std():.3f}",
            f"{f1.mean():.3f}±{f1.std():.3f}",
            f"{auc.mean():.3f}±{auc.std():.3f}" if not np.isnan(auc).all() else "  N/A",
        ))
    print("=" * 55)


def save_results_csv(results: dict, path: str):
    rows = []
    for model_name, folds in results.items():
        for fold_idx, (acc, f1, auc) in enumerate(
            zip(folds['accuracy'], folds['f1_score'], folds['roc_auc'])
        ):
            rows.append({
                'model':    model_name,
                'fold':     fold_idx + 1,
                'accuracy': round(acc, 4),
                'f1_score': round(f1,  4),
                'roc_auc':  round(auc, 4) if not np.isnan(auc) else None,
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"\nCV results saved to: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def evaluate_models():
    # ── 1. Load data ─────────────────────────────────────────────────────────
    print("Loading data...")
    X, y, file_ids = load_or_extract_features(TRAIN_AUDIO_DIR, cache_path=CACHE_FILE)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features | "
          f"AD={y.sum()}, CN={(y == 0).sum()}")

    # ── 2. Prepare CV splits ─────────────────────────────────────────────────
    splits = get_kfold_splits(X, y, n_splits=N_SPLITS)

    results = {m: {'accuracy': [], 'f1_score': [], 'roc_auc': []}
               for m in MODELS_TO_TEST}

    # ── 3. Cross-validation loop ─────────────────────────────────────────────
    print(f"\nRunning {N_SPLITS}-Fold Stratified CV over "
          f"{len(MODELS_TO_TEST)} models...\n")

    for fold, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"── Fold {fold + 1}/{N_SPLITS} "
              f"(train={len(train_idx)}, val={len(val_idx)}) ──")

        for model_name in MODELS_TO_TEST:
            pipeline = ModelFactory.get_model(model_name)
            acc, f1, auc = evaluate_pipeline(pipeline, X_train, y_train, X_val, y_val)

            results[model_name]['accuracy'].append(acc)
            results[model_name]['f1_score'].append(f1)
            results[model_name]['roc_auc'].append(auc)

            auc_str = f"{auc:.4f}" if not np.isnan(auc) else "N/A"
            print(f"  [{model_name:<22}]  "
                  f"Acc={acc:.4f}  F1={f1:.4f}  AUC={auc_str}")
        print()

    # ── 4. Summary & save ────────────────────────────────────────────────────
    print_summary(results)
    save_results_csv(results, OUTPUT_CSV)


if __name__ == "__main__":
    evaluate_models()