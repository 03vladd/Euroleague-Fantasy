"""
Train ML model to predict next round fantasy points
Uses XGBoost with time-series cross-validation
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
from pathlib import Path

# Load tuned hyperparameters if tune_model.py has been run, otherwise use defaults
_params_path = Path('models/best_params.json')
if _params_path.exists():
    with open(_params_path) as f:
        BEST_PARAMS = json.load(f)
    print(f"Loaded tuned hyperparameters from {_params_path}")
else:
    BEST_PARAMS = {
        'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.02,
        'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5,
        'gamma': 0.1, 'reg_alpha': 0.05, 'reg_lambda': 1.0,
    }
    print("No tuned params found — using defaults. Run tune_model.py to optimise.")

print("=" * 60)
print("FANTASY POINTS PREDICTION MODEL")
print("=" * 60)

# Load game-by-game data with features
df = pd.read_csv('data/processed/player_games_with_features.csv')

# Filter to players with reasonable playing time
df = df[df['Minutes'] >= 5].copy()
print(f"\nGames with 5+ minutes: {len(df)}")

# Create target: next game fantasy points (requires Player+Round ordering)
df = df.sort_values(['Player', 'Round']).reset_index(drop=True)
df['target'] = df.groupby('Player')['fantasy_points'].shift(-1)

# Remove last game for each player (no target available)
df = df[df['target'].notna()].copy()

# Re-sort by Round so TimeSeriesSplit folds split by time, not alphabet
df = df.sort_values('Round').reset_index(drop=True)
print(f"Training samples: {len(df)}")

print("\n" + "=" * 60)
print("FEATURE SELECTION")
print("=" * 60)

# Select features for prediction
feature_cols = [
    # Recent performance
    'fp_last_3', 'fp_last_5', 'fp_last_10',
    'fp_std_3', 'fp_std_5', 'fp_std_10',

    # Minutes / usage
    'Minutes', 'minutes_last_3', 'minutes_last_5', 'minutes_last_10',
    'minutes_rank', 'usage_trend',

    # Role stability
    'starter_rate_3', 'starter_rate_5',

    # Team quality when on court
    'plusminus_last_3', 'plusminus_last_5',

    # Form
    'form_trend', 'hot_streak',
    'points_share',

    # Opponent & game context
    'opp_defensive_rating',
    'team_won', 'Home',

    # Risk & home tendency
    'dnp_rate_last5', 'dnp_rate_last10', 'home_advantage',

    # Experience
    'games_played',

    # Last game box score
    'Points', 'TotalRebounds', 'Assistances', 'Valuation'
]

X = df[feature_cols].copy()
y = df['target'].copy()

print(f"Features: {len(feature_cols)}")
print(f"Samples: {len(X)}")

print("\n" + "=" * 60)
print("TIME-SERIES CROSS-VALIDATION")
print("=" * 60)

# Use time-series split (respect temporal order)
tscv = TimeSeriesSplit(n_splits=5)

cv_scores = []
fold = 1

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train XGBoost
    model = xgb.XGBRegressor(**BEST_PARAMS, random_state=42, n_jobs=-1, verbosity=0)

    model.fit(X_train, y_train, verbose=False)

    # Predict
    y_pred = model.predict(X_val)

    # Evaluate
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    cv_scores.append({'fold': fold, 'mae': mae, 'rmse': rmse, 'r2': r2})
    print(f"Fold {fold}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
    fold += 1

print(f"\n{'=' * 60}")
print("AVERAGE CROSS-VALIDATION SCORES")
print(f"{'=' * 60}")

cv_df = pd.DataFrame(cv_scores)
print(f"MAE:  {cv_df['mae'].mean():.2f} ± {cv_df['mae'].std():.2f}")
print(f"RMSE: {cv_df['rmse'].mean():.2f} ± {cv_df['rmse'].std():.2f}")
print(f"R²:   {cv_df['r2'].mean():.3f} ± {cv_df['r2'].std():.3f}")

print(f"\n{'=' * 60}")
print("TRAINING FINAL MODEL")
print(f"{'=' * 60}")

# Train on all data using the same tuned params as CV
final_model = xgb.XGBRegressor(**BEST_PARAMS, random_state=42, n_jobs=-1, verbosity=0)

final_model.fit(X, y, verbose=False)

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance.head(10).to_string(index=False))

# Save model
Path('models').mkdir(exist_ok=True)
with open('models/fantasy_predictor.pkl', 'wb') as f:
    pickle.dump(final_model, f)

# Save feature list
with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)

print(f"\n✓ Model saved to: models/fantasy_predictor.pkl")

print(f"\n{'=' * 60}")
print("GENERATING PREDICTIONS FOR NEXT ROUND")
print(f"{'=' * 60}")

# Load latest player data
latest = pd.read_csv('data/processed/prediction_features.csv')

# Filter to players with prices
latest = latest[latest['price'].notna()].copy()

# Ensure all features exist
for col in feature_cols:
    if col not in latest.columns:
        print(f"Warning: Missing feature {col}, setting to 0")
        latest[col] = 0

# Predict
X_pred = latest[feature_cols]
latest['predicted_fp'] = final_model.predict(X_pred)

# Calculate value
latest['predicted_value'] = latest['predicted_fp'] / latest['price']

# Save predictions
extra_cols = [c for c in ['dnp_rate_last5', 'dnp_rate_last10', 'home_advantage']
              if c in latest.columns]
latest_relevant = latest[[
    'Player', 'Team', 'price',
    'fantasy_points_mean', 'fp_last_3', 'fp_last_5',
    'predicted_fp', 'predicted_value',
    'games_played_max',
] + extra_cols].copy()

latest_relevant.to_csv('data/processed/next_round_predictions.csv', index=False)

print(f"\n✓ Predictions saved to: data/processed/next_round_predictions.csv")

print(f"\n{'=' * 60}")
print("TOP 20 PREDICTED PERFORMERS (Next Round)")
print(f"{'=' * 60}")

top_pred = latest_relevant.nlargest(20, 'predicted_fp')
print(f"\n{'Player':<25} {'Team':<5} {'Price':>6} {'Last3':>6} {'Pred':>6}")
print("-" * 60)
for _, p in top_pred.iterrows():
    print(f"{p['Player']:<25} {p['Team']:<5} {p['price']:>6.1f} "
          f"{p['fp_last_3']:>6.1f} {p['predicted_fp']:>6.1f}")

print(f"\n{'=' * 60}")
print("TOP 20 BEST VALUE (Predicted Points per Credit)")
print(f"{'=' * 60}")

top_value = latest_relevant.nlargest(20, 'predicted_value')
print(f"\n{'Player':<25} {'Team':<5} {'Price':>6} {'Pred':>6} {'Value':>6}")
print("-" * 60)
for _, p in top_value.iterrows():
    print(f"{p['Player']:<25} {p['Team']:<5} {p['price']:>6.1f} "
          f"{p['predicted_fp']:>6.1f} {p['predicted_value']:>6.3f}")

print(f"\n{'=' * 60}")
print("DONE! Next: Run optimized_team_v2.py")
print(f"{'=' * 60}")