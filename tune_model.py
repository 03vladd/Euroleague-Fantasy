"""
Hyperparameter tuning + feature importance analysis.

Run this once before train_model.py to find the best XGBoost configuration.
Saves models/best_params.json — train_model.py loads it automatically.

Usage:
    python tune_model.py
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shaphow
import xgboost as xgb
import optuna
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

import sys
SKIP_TUNING = '--skip-tuning' in sys.argv or Path('models/best_params.json').exists()

# ── Data (identical pre-processing to train_model.py) ────────
df = pd.read_csv('data/processed/player_games_with_features.csv')
df = df[df['Minutes'] >= 5].copy()
df = df.sort_values(['Player', 'Round']).reset_index(drop=True)
df['target'] = df.groupby('Player')['fantasy_points'].shift(-1)
df = df[df['target'].notna()].copy()
df = df.sort_values('Round').reset_index(drop=True)

with open('models/feature_columns.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

for col in feature_cols:
    if col not in df.columns:
        print(f"  Warning: '{col}' missing from data — filling with 0")
        df[col] = 0

X = df[feature_cols].fillna(0)
y = df['target']

tscv = TimeSeriesSplit(n_splits=5)

# ═══════════════════════════════════════════════════════════════
# PART 1 — HYPERPARAMETER OPTIMISATION
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("PART 1 — HYPERPARAMETER OPTIMISATION (Optuna)")
print("=" * 60)
print("""
Each trial trains 5 time-series CV folds and returns average MAE.
Optuna uses Bayesian optimisation (Tree-structured Parzen Estimator)
to focus on promising regions of the search space rather than
brute-force grid search.
""")

def objective(trial: optuna.Trial) -> float:
    params = {
        # How many trees to build. More trees = more capacity.
        # Paired with learning_rate — lower rate needs more trees.
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),

        # Max depth of each tree. Deeper = more complex rules but
        # more likely to overfit individual players.
        'max_depth': trial.suggest_int('max_depth', 3, 6),

        # How aggressively each tree corrects the previous one.
        # Log scale because small differences matter near 0.
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),

        # Fraction of rows randomly sampled per tree.
        # <1.0 introduces variance and prevents memorisation.
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),

        # Fraction of features randomly sampled per tree.
        # Keeps each tree from always leaning on the same features.
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),

        # Minimum number of samples required in a leaf.
        # High values stop the model from overfitting tiny player groups.
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),

        # Minimum gain a split must produce to be accepted.
        # Extra pruning on top of min_child_weight.
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),

        # L1 regularisation — pushes less useful leaf weights toward zero.
        # Acts like soft feature selection inside the model.
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),

        # L2 regularisation — keeps all leaf weights smooth and small.
        # Prevents any single tree from making extreme corrections.
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),

        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }

    fold_maes = []
    for train_idx, val_idx in tscv.split(X):
        model = xgb.XGBRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)
        preds = model.predict(X.iloc[val_idx])
        fold_maes.append(mean_absolute_error(y.iloc[val_idx], preds))

    return float(np.mean(fold_maes))


if SKIP_TUNING:
    print("Tuned params already exist — skipping Optuna (pass --force to re-run).")
    with open('models/best_params.json') as f:
        best_params = json.load(f)
    print(f"Loaded: {best_params}")
else:
    N_TRIALS = 60
    print(f"Running {N_TRIALS} trials — estimated time: 3-5 minutes\n")

    study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_mae    = study.best_value

    print(f"\nBest CV MAE:  {best_mae:.3f}")
    print(f"Best params:")
    for k, v in best_params.items():
        print(f"  {k:<22} {v}")

    Path('models').mkdir(exist_ok=True)
    with open('models/best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"\n✓ Saved to models/best_params.json")
    print("  train_model.py will load these automatically on next run.")

# ═══════════════════════════════════════════════════════════════
# PART 2 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 2 — FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)
print("""
Three methods, each measuring something different:

  XGB Gain       — average improvement in prediction error each time
                   a feature is used for a split. Fast but can favour
                   high-cardinality features.

  Permutation    — how much MAE rises when a feature's values are
                   randomly shuffled. Model-agnostic and directly
                   measures predictive contribution on held-out data.
                   Negative = feature adds noise, not signal.

  SHAP           — explains each individual prediction: "this feature
                   pushed the prediction up/down by X points." The
                   mean |SHAP| across all samples gives a fair,
                   consistent importance score.
""")

# Train on first 80% (rounds), evaluate on last 20%
split = int(len(X) * 0.8)
final_params = {**best_params, 'random_state': 42, 'n_jobs': -1, 'verbosity': 0}
model = xgb.XGBRegressor(**final_params)
model.fit(X.iloc[:split], y.iloc[:split], verbose=False)

X_val = X.iloc[split:].reset_index(drop=True)
y_val = y.iloc[split:].reset_index(drop=True)

# ── XGBoost gain importance ───────────────────────────────────
booster    = model.get_booster()
gain_raw   = booster.get_score(importance_type='gain')
gain_vals  = [gain_raw.get(f, 0.0) for f in feature_cols]

# ── Permutation importance ────────────────────────────────────
print("Computing permutation importance (10 repeats)…")
perm = permutation_importance(
    model, X_val, y_val,
    n_repeats=10, random_state=42, n_jobs=-1,
    scoring='neg_mean_absolute_error'
)
# permutation_importance returns negative MAE change; flip sign so
# positive = feature helps, negative = feature hurts
perm_mean = -perm.importances_mean   # higher = more important
perm_std  =  perm.importances_std

# ── SHAP values ───────────────────────────────────────────────
print("Computing SHAP values…")
shap_imp = None
try:
    # XGBoost 3.x stores base_score as '[value]' string; patch SHAP's float
    import shap.explainers._tree as _shap_tree
    _builtin_float = float
    def _xgb3_safe_float(x):
        if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
            return _builtin_float(x[1:-1])
        return _builtin_float(x)
    _shap_tree.float = _xgb3_safe_float
    explainer   = shap.TreeExplainer(model)
    shap_matrix = explainer.shap_values(X_val)
    del _shap_tree.float
    shap_imp = np.abs(shap_matrix).mean(axis=0)
    print("✓ SHAP values computed")
except Exception as e:
    print(f"⚠ SHAP skipped ({type(e).__name__}: {e})")
    print("  This is a known XGBoost 3.x / SHAP incompatibility.")
    shap_imp = np.zeros(len(feature_cols))

# ── Combined table ────────────────────────────────────────────
imp_df = pd.DataFrame({
    'feature':       feature_cols,
    'xgb_gain':      gain_vals,
    'perm_imp':      perm_mean,
    'perm_std':      perm_std,
    'shap_imp':      shap_imp,
})

# Rank by permutation importance if SHAP failed, otherwise SHAP
rank_col = 'shap_imp' if shap_imp.any() else 'perm_imp'
imp_df = imp_df.sort_values(rank_col, ascending=False).reset_index(drop=True)
imp_df['rank'] = imp_df.index + 1

print(f"\n{'Rk':<4} {'Feature':<25} {'XGB Gain':>10} {'Perm Imp':>10} {'±':>6} {'SHAP':>8}")
print("─" * 68)
for _, row in imp_df.iterrows():
    flag = "  ← drop?" if row['perm_imp'] < 0 else ""
    print(f"{int(row['rank']):<4} {row['feature']:<25} "
          f"{row['xgb_gain']:>10.1f} "
          f"{row['perm_imp']:>10.4f} "
          f"±{row['perm_std']:>5.4f} "
          f"{row['shap_imp']:>8.4f}{flag}")

# ── Candidates to drop ────────────────────────────────────────
to_drop = imp_df[imp_df['perm_imp'] < 0]['feature'].tolist()
print()
if to_drop:
    print(f"Features with negative permutation importance (adding noise):")
    for f in to_drop:
        print(f"  ✗  {f}")
    print("\n  Consider removing these from feature_cols in train_model.py")
else:
    print("All features have positive permutation importance — none to drop.")

imp_df.to_csv('models/feature_importance.csv', index=False)
print(f"\n✓ Full importance table saved to models/feature_importance.csv")
print("\nNext step: run train_model.py — it will use the tuned parameters.")