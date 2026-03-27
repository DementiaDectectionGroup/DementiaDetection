"""
DementiaPred – Phase 2 main entry point
========================================
Phase 2 adds:
  - balanced_accuracy + macro_f1 metrics
  - GridSearchCV tuning for logistic_regression and svm_linear
  - Optional feature reduction: none | pca | selectkbest (applied inside pipeline)
  - cv_summary.csv  (mean ± std per configuration, one row each)
  - Ranked summary console table

Usage:
    python main.py                        # all models, no reduction, with tuning
    python main.py --reduce pca           # all tunable models with PCA
    python main.py --reduce selectkbest   # all tunable models with SelectKBest
    python main.py --no-tune              # skip GridSearchCV (faster)
    python main.py --models lr svm_linear # run subset only
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score,
)

from dataprocess import load_or_extract_features, get_kfold_splits
from modelfactory import ModelFactory

# ── Configuration ─────────────────────────────────────────────────────────────
TRAIN_AUDIO_DIR = "ADReSSo/train/audio"
CACHE_FILE      = "train_features_cache.npz"
N_SPLITS        = 5
N_COMPONENTS    = 30          # features kept by PCA / SelectKBest
FOLD_CSV        = "cv_results.csv"
SUMMARY_CSV     = "cv_summary.csv"

# Short aliases accepted on the CLI
_MODEL_ALIASES = {
    'lr':                  'logistic_regression',
    'logistic_regression': 'logistic_regression',
    'svm_linear':          'svm_linear',
    'svm_rbf':             'svm_rbf',
    'rf':                  'random_forest',
    'random_forest':       'random_forest',
    'gb':                  'gradient_boosting',
    'gradient_boosting':   'gradient_boosting',
    'majority':            'majority',
}

# Reduction modes to sweep when --reduce is given
REDUCTION_MODES = ['none', 'pca', 'selectkbest']


# ── Metrics ───────────────────────────────────────────────────────────────────
def evaluate_pipeline(pipeline, X_train, y_train, X_val, y_val) -> dict:
    """
    Fit pipeline on training fold and evaluate on validation fold.
    Returns a dict of all Phase-2 metrics.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    acc      = accuracy_score(y_val, y_pred)
    bal_acc  = balanced_accuracy_score(y_val, y_pred)
    f1_bin   = f1_score(y_val, y_pred, zero_division=0)            # binary (positive class)
    f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)

    # predict_proba is present on GridSearchCV-wrapped pipelines too
    if hasattr(pipeline, 'predict_proba'):
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        auc    = roc_auc_score(y_val, y_prob)
    else:
        auc = float('nan')

    return dict(accuracy=acc, balanced_accuracy=bal_acc,
                f1_score=f1_bin, macro_f1=f1_macro, roc_auc=auc)


# ── Console output ────────────────────────────────────────────────────────────
def print_ranked_summary(summary_df: pd.DataFrame):
    """Print summary table ranked by mean ROC-AUC descending."""
    cols   = ['config', 'acc_mean', 'bal_acc_mean', 'f1_mean', 'macro_f1_mean', 'auc_mean']
    ranked = summary_df.sort_values('auc_mean', ascending=False).reset_index(drop=True)

    width = 42
    sep   = "=" * (width + 52)
    hfmt  = f"{{:<{width}}}  {{:>8}}  {{:>8}}  {{:>8}}  {{:>8}}  {{:>8}}"
    rfmt  = f"{{:<{width}}}  {{:>8}}  {{:>8}}  {{:>8}}  {{:>8}}  {{:>8}}"

    print(f"\n{sep}")
    print(f"{'Phase 2 – Cross-Validation Summary  (ranked by AUC)':^{width + 52}}")
    print(sep)
    print(hfmt.format("Configuration", "Acc", "BalAcc", "F1", "MacroF1", "AUC"))
    print("-" * (width + 52))

    for _, row in ranked.iterrows():
        def fmt(m, s): return f"{m:.3f}±{s:.3f}"
        print(rfmt.format(
            row['config'],
            fmt(row['acc_mean'],      row['acc_std']),
            fmt(row['bal_acc_mean'],  row['bal_acc_std']),
            fmt(row['f1_mean'],       row['f1_std']),
            fmt(row['macro_f1_mean'], row['macro_f1_std']),
            fmt(row['auc_mean'],      row['auc_std']) if not np.isnan(row['auc_mean']) else "   N/A",
        ))

    print(sep)
    best = ranked.iloc[0]
    print(f"\n★  Best config : {best['config']}")
    print(f"   ROC-AUC     : {best['auc_mean']:.4f} ± {best['auc_std']:.4f}")
    print(f"   Bal-Acc     : {best['bal_acc_mean']:.4f} ± {best['bal_acc_std']:.4f}\n")


# ── CSV helpers ───────────────────────────────────────────────────────────────
def build_fold_rows(config_name: str, fold_results: list[dict]) -> list[dict]:
    rows = []
    for fold_idx, metrics in enumerate(fold_results):
        row = {'config': config_name, 'fold': fold_idx + 1}
        row.update({k: round(v, 4) if not np.isnan(v) else None
                    for k, v in metrics.items()})
        rows.append(row)
    return rows


def build_summary_row(config_name: str, fold_results: list[dict]) -> dict:
    keys = ['accuracy', 'balanced_accuracy', 'f1_score', 'macro_f1', 'roc_auc']
    row  = {'config': config_name}
    for k in keys:
        vals = np.array([f[k] for f in fold_results], dtype=float)
        short = (k.replace('accuracy', 'acc')
                  .replace('balanced_acc', 'bal_acc')
                  .replace('f1_score', 'f1')
                  .replace('macro_f1', 'macro_f1')
                  .replace('roc_auc', 'auc'))
        row[f'{short}_mean'] = round(float(np.nanmean(vals)), 4)
        row[f'{short}_std']  = round(float(np.nanstd(vals)),  4)
    return row


# ── Core CV loop ──────────────────────────────────────────────────────────────
def run_cv(X, y, splits, model_name: str, reduction: str,
           tune: bool) -> list[dict]:
    """Run one full CV for a (model, reduction) config. Returns per-fold metrics."""
    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if tune and ModelFactory.has_param_grid(model_name):
            pipeline = ModelFactory.get_tuned_model(
                model_name, reduction=reduction,
                n_components=N_COMPONENTS, cv=3, scoring='roc_auc',
            )
        else:
            pipeline = ModelFactory.get_model(
                model_name, reduction=reduction, n_components=N_COMPONENTS,
            )

        metrics = evaluate_pipeline(pipeline, X_tr, y_tr, X_val, y_val)
        fold_results.append(metrics)

        auc_str = f"{metrics['roc_auc']:.4f}" if not np.isnan(metrics['roc_auc']) else "N/A"
        print(f"    fold {fold+1}  acc={metrics['accuracy']:.4f}  "
              f"bal={metrics['balanced_accuracy']:.4f}  "
              f"f1={metrics['f1_score']:.4f}  "
              f"mf1={metrics['macro_f1']:.4f}  auc={auc_str}")

    return fold_results


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DementiaPred Phase 2")
    p.add_argument(
        '--reduce', choices=['none', 'pca', 'selectkbest', 'all'],
        default='none',
        help="Feature reduction mode (default: none). Use 'all' to sweep all modes.",
    )
    p.add_argument(
        '--models', nargs='+', default=None,
        metavar='MODEL',
        help="Model(s) to run (default: all). Aliases: lr, rf, gb, ...",
    )
    p.add_argument(
        '--no-tune', dest='tune', action='store_false',
        help="Disable GridSearchCV tuning (faster).",
    )
    p.add_argument(
        '--tune-only', nargs='+', default=None,
        metavar='MODEL',
        help="Restrict tuning to specific models only.",
    )
    return p.parse_args()


def resolve_models(raw_list) -> list[str]:
    if raw_list is None:
        return ModelFactory.all_model_names()
    resolved = []
    for m in raw_list:
        key = _MODEL_ALIASES.get(m.lower())
        if key is None:
            raise ValueError(f"Unknown model alias '{m}'. "
                             f"Valid: {list(_MODEL_ALIASES.keys())}")
        if key not in resolved:
            resolved.append(key)
    return resolved


def evaluate_models():
    args = parse_args()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading data...")
    X, y, _ = load_or_extract_features(TRAIN_AUDIO_DIR, cache_path=CACHE_FILE)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features | "
          f"AD={int(y.sum())}, CN={int((y == 0).sum())}")

    splits      = get_kfold_splits(X, y, n_splits=N_SPLITS)
    models      = resolve_models(args.models)
    reductions  = REDUCTION_MODES if args.reduce == 'all' else [args.reduce]
    tune        = args.tune

    print(f"\nModels    : {models}")
    print(f"Reductions: {reductions}")
    print(f"Tuning    : {'enabled (GridSearchCV inner-CV=3)' if tune else 'disabled'}")
    print(f"Folds     : {N_SPLITS}\n")

    all_fold_rows    = []
    all_summary_rows = []

    # ── 2. Outer CV loop ──────────────────────────────────────────────────────
    for model_name in models:
        for reduction in reductions:
            # Skip reduction for non-numeric-scale models (majority baseline)
            if model_name == 'majority' and reduction != 'none':
                continue

            config_name = (f"{model_name}+{reduction}"
                           if reduction != 'none' else model_name)
            print(f"── {config_name} {'[tuned]' if tune and ModelFactory.has_param_grid(model_name) else ''} ──")

            fold_results = run_cv(X, y, splits, model_name, reduction, tune)

            all_fold_rows.extend(build_fold_rows(config_name, fold_results))
            all_summary_rows.append(build_summary_row(config_name, fold_results))
            print()

    # ── 3. Persist results ────────────────────────────────────────────────────
    fold_df    = pd.DataFrame(all_fold_rows)
    summary_df = pd.DataFrame(all_summary_rows)

    fold_df.to_csv(FOLD_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Fold-level results → {FOLD_CSV}")
    print(f"Summary results    → {SUMMARY_CSV}")

    # ── 4. Ranked summary table ───────────────────────────────────────────────
    print_ranked_summary(summary_df)

    # ── 5. Best config per base model ─────────────────────────────────────────
    print("Best configuration per base model (by mean AUC):")
    print("-" * 50)
    summary_df['base_model'] = summary_df['config'].str.split('+').str[0]
    for base, grp in summary_df.groupby('base_model'):
        best = grp.loc[grp['auc_mean'].idxmax()]
        print(f"  {base:<25} → {best['config']:<35}  AUC={best['auc_mean']:.4f}")
    print()


if __name__ == "__main__":
    evaluate_models()