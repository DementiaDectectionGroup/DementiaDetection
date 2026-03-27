"""
DementiaPred – Phase 3 main entry point
========================================
Phase 3 adds:
  - Prosodic + fluency feature groups (F0, voiced ratio, pause stats, speech-rate proxy)
  - Named feature groups with ablation presets
  - --features / --ablation CLI flags
  - All Phase 2 functionality preserved

Feature presets (--features):
    all                    all 7 groups (625 + 15 new = 640 dims)
    baseline_acoustic      mfcc + delta_mfcc + delta2_mfcc + spectral + energy  (625)
    prosody_only           prosodic (11)
    fluency_only           fluency (4)
    baseline_plus_prosody  baseline + prosodic (636)
    baseline_plus_fluency  baseline + fluency (629)
    prosody_and_fluency    prosodic + fluency (15)

Usage:
    python main.py                                    # all models, baseline_acoustic features
    python main.py --features all                     # all feature groups
    python main.py --features prosody_only            # prosodic features only
    python main.py --ablation                         # full feature-group ablation (lr + svm_linear)
    python main.py --models lr --features all --no-tune
    python main.py --ablation --models lr             # ablation with lr only
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, roc_auc_score,
)

from dataprocess import (
    load_or_extract_features, get_kfold_splits,
    select_features, FEATURE_PRESETS, ALL_GROUPS,
)
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

# Models used in the ablation sweep
ABLATION_MODELS = ['logistic_regression', 'svm_linear']

# Ablation presets run in order
ABLATION_PRESETS = [
    'baseline_acoustic',
    'prosody_only',
    'fluency_only',
    'prosody_and_fluency',
    'baseline_plus_prosody',
    'baseline_plus_fluency',
    'all',
]


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
def print_ranked_summary(summary_df: pd.DataFrame, title: str = "Cross-Validation Summary"):
    ranked = summary_df.sort_values('auc_mean', ascending=False).reset_index(drop=True)
    width  = 44
    sep    = "=" * (width + 52)
    hfmt   = f"{{:<{width}}}  {{:>8}}  {{:>8}}  {{:>8}}  {{:>8}}  {{:>8}}"

    print(f"\n{sep}")
    print(f"{title + '  (ranked by AUC)':^{width + 52}}")
    print(sep)
    print(hfmt.format("Configuration", "Acc", "BalAcc", "F1", "MacroF1", "AUC"))
    print("-" * (width + 52))

    for _, row in ranked.iterrows():
        def fmt(m, s): return f"{m:.3f}±{s:.3f}"
        nan_auc = np.isnan(row['auc_mean'])
        print(hfmt.format(
            str(row['config'])[:width],
            fmt(row['acc_mean'],      row['acc_std']),
            fmt(row['bal_acc_mean'],  row['bal_acc_std']),
            fmt(row['f1_mean'],       row['f1_std']),
            fmt(row['macro_f1_mean'], row['macro_f1_std']),
            fmt(row['auc_mean'],      row['auc_std']) if not nan_auc else "     N/A",
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
    _rename = {
        'accuracy':           'acc',
        'balanced_accuracy':  'bal_acc',
        'f1_score':           'f1',
        'macro_f1':           'macro_f1',
        'roc_auc':            'auc',
    }
    row = {'config': config_name}
    for k, short in _rename.items():
        vals = np.array([f[k] for f in fold_results], dtype=float)
        row[f'{short}_mean'] = round(float(np.nanmean(vals)), 4)
        row[f'{short}_std']  = round(float(np.nanstd(vals)),  4)
    return row


# ── Core CV loop ──────────────────────────────────────────────────────────────
def run_cv(X: np.ndarray, y: np.ndarray, splits: list,
           model_name: str, reduction: str, tune: bool) -> list[dict]:
    """Single (model, reduction, feature-set) CV run. Returns per-fold metric dicts."""
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


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="DementiaPred Phase 3")

    p.add_argument(
        '--features',
        choices=list(FEATURE_PRESETS.keys()),
        default='baseline_acoustic',
        help="Feature preset (default: baseline_acoustic). Ignored when --ablation is set.",
    )
    p.add_argument(
        '--ablation', action='store_true',
        help="Run full feature-group ablation sweep (overrides --features).",
    )
    p.add_argument(
        '--reduce', choices=['none', 'pca', 'selectkbest', 'all'],
        default='none',
        help="Feature reduction mode (default: none).",
    )
    p.add_argument(
        '--models', nargs='+', default=None, metavar='MODEL',
        help="Model(s) to run. Aliases: lr, rf, gb, ... Default: all (or ablation models).",
    )
    p.add_argument(
        '--no-tune', dest='tune', action='store_false',
        help="Disable GridSearchCV tuning.",
    )
    return p.parse_args()


def resolve_models(raw_list) -> list[str]:
    if raw_list is None:
        return ModelFactory.all_model_names()
    resolved = []
    for m in raw_list:
        key = _MODEL_ALIASES.get(m.lower())
        if key is None:
            raise ValueError(f"Unknown model alias '{m}'. Valid: {list(_MODEL_ALIASES.keys())}")
        if key not in resolved:
            resolved.append(key)
    return resolved


# ── Main ──────────────────────────────────────────────────────────────────────
def evaluate_models():
    args = parse_args()

    # ── 1. Load grouped feature arrays ───────────────────────────────────────
    print("Loading data...")
    group_arrays, y, _ = load_or_extract_features(TRAIN_AUDIO_DIR, cache_path=CACHE_FILE)

    # Report dimensionalities
    dim_info = {g: group_arrays[g].shape[1] for g in ALL_GROUPS}
    print(f"Dataset: {y.shape[0]} samples | AD={int(y.sum())}, CN={int((y==0).sum())}")
    print(f"Feature groups: { {g: d for g, d in dim_info.items()} }")

    tune = args.tune

    # ── 2. Build experiment list ──────────────────────────────────────────────
    # Each experiment: (config_label, X, model_name, reduction)
    experiments: list[tuple[str, np.ndarray, str, str]] = []

    if args.ablation:
        # Feature-group ablation with ABLATION_MODELS (or user subset)
        models     = resolve_models(args.models) if args.models else ABLATION_MODELS
        reductions = ['none']   # ablation: no reduction to isolate feature effect
        print(f"\nAblation mode | models={models}")

        for preset_name in ABLATION_PRESETS:
            groups_sel = FEATURE_PRESETS[preset_name]
            X = select_features(group_arrays, groups_sel)
            n_feat = X.shape[1]
            for model_name in models:
                label = f"{model_name}|{preset_name}({n_feat}d)"
                experiments.append((label, X, model_name, 'none'))
    else:
        # Normal mode: single feature preset, user-selected models + reductions
        models     = resolve_models(args.models)
        reductions = REDUCTION_MODES if args.reduce == 'all' else [args.reduce]
        preset     = args.features
        groups_sel = FEATURE_PRESETS[preset]
        X          = select_features(group_arrays, groups_sel)
        print(f"\nNormal mode | features={preset} ({X.shape[1]}d) | "
              f"models={models} | reductions={reductions}")

        for model_name in models:
            for reduction in reductions:
                if model_name == 'majority' and reduction != 'none':
                    continue
                label = (f"{model_name}+{reduction}|{preset}"
                         if reduction != 'none' else f"{model_name}|{preset}")
                experiments.append((label, X, model_name, reduction))

    print(f"Tuning : {'enabled (inner CV=3)' if tune else 'disabled'}")
    print(f"Total experiments: {len(experiments)}\n")

    # ── 3. Run all experiments ────────────────────────────────────────────────
    all_fold_rows:    list[dict] = []
    all_summary_rows: list[dict] = []

    for config_name, X_exp, model_name, reduction in experiments:
        # Recompute splits per-experiment so stratification is consistent
        splits = get_kfold_splits(X_exp, y, n_splits=N_SPLITS)
        tuning_flag = tune and not args.ablation   # keep ablation fast by default
        # allow tuning in ablation only when explicitly requested via --tune flag
        # (ablation with tuning is slow; user can opt-in by omitting --no-tune)
        tuning_flag = tune

        is_tuned = tune and ModelFactory.has_param_grid(model_name)
        print(f"── {config_name} {'[tuned]' if is_tuned else ''} ──")
        fold_results = run_cv(X_exp, y, splits, model_name, reduction, tune)
        all_fold_rows.extend(build_fold_rows(config_name, fold_results))
        all_summary_rows.append(build_summary_row(config_name, fold_results))
        print()

    # ── 4. Persist results ────────────────────────────────────────────────────
    fold_df    = pd.DataFrame(all_fold_rows)
    summary_df = pd.DataFrame(all_summary_rows)

    fold_df.to_csv(FOLD_CSV, index=False)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Fold-level results → {FOLD_CSV}")
    print(f"Summary results    → {SUMMARY_CSV}")

    # ── 5. Ranked summary ─────────────────────────────────────────────────────
    title = "Phase 3 – Ablation" if args.ablation else "Phase 3 – CV Summary"
    print_ranked_summary(summary_df, title=title)

    # ── 6. Best config per base model ─────────────────────────────────────────
    print("Best configuration per base model (by mean AUC):")
    print("-" * 60)
    summary_df['base_model'] = summary_df['config'].str.split('[|+]', regex=True).str[0]
    for base, grp in summary_df.groupby('base_model'):
        valid = grp.dropna(subset=['auc_mean'])
        if valid.empty:
            continue
        best = valid.loc[valid['auc_mean'].idxmax()]
        print(f"  {base:<25} → {str(best['config']):<45}  AUC={best['auc_mean']:.4f}")
    print()


if __name__ == "__main__":
    evaluate_models()