"""
backtest.py — Round-by-round backtest of ML fantasy team predictions.

For each round R from START_ROUND to the last round:
  1. Train XGBoost on all player-game rows from rounds 1..R-1
  2. For each player, take their most recent features from rounds <R to predict round R
  3. Optimize a team via LP (4G/4F/2C, budget 100, max 3/team, 1 captain 2×)
  4. Score predicted team on actual round R fantasy points
     (captain = whoever we predicted highest on that team)
  5. Compare vs:
     - Oracle constrained: LP on actual FP (best possible team with constraints)
     - Oracle unconstrained: top-10 players by actual FP (no budget/position limits)
     - Naive baseline: top players by fp_last_3 (rolling 3-game avg) with LP

Output: backtest_results.csv + summary printed to stdout.
"""

import json
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
START_ROUND = 6      # need enough history for rolling features to be meaningful
BUDGET      = 100
ROSTER      = 10
MAX_PER_TEAM = 3
POSITION_REQ = {'G': 4, 'F': 4, 'C': 2}
POSITION_MAP = {'Guard': 'G', 'Forward': 'F', 'Center': 'C'}

FEATURE_COLS = [
    'fp_last_3', 'fp_last_5', 'fp_last_10',
    'fp_std_3', 'fp_std_5', 'fp_std_10',
    'Minutes', 'minutes_last_3', 'minutes_last_5', 'minutes_last_10',
    'minutes_rank', 'usage_trend',
    'starter_rate_3', 'starter_rate_5',
    'plusminus_last_3', 'plusminus_last_5',
    'form_trend', 'hot_streak',
    'points_share',
    'opp_defensive_rating',
    'team_won', 'Home',
    'games_played',
    'Points', 'TotalRebounds', 'Assistances', 'Valuation',
]

# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("EUROLEAGUE FANTASY BACKTEST")
print("=" * 65)

df = pd.read_csv('data/processed/player_games_with_features.csv')
df = df[df['Minutes'] >= 5].copy()
df = df.sort_values(['Player', 'Round']).reset_index(drop=True)
print(f"Loaded {len(df):,} rows  |  Rounds {df['Round'].min()}–{df['Round'].max()}")

# Load params
params_path = Path('models/best_params.json')
if params_path.exists():
    with open(params_path) as f:
        BEST_PARAMS = json.load(f)
    print(f"Using Optuna-tuned params from {params_path}")
else:
    BEST_PARAMS = {
        'n_estimators': 400, 'max_depth': 4, 'learning_rate': 0.02,
        'subsample': 0.8, 'colsample_bytree': 0.7, 'min_child_weight': 5,
        'gamma': 0.1, 'reg_alpha': 0.05, 'reg_lambda': 1.0,
    }
    print("Using default params (run tune_model.py to improve)")

# Load prices
prices = pd.read_csv('euroleague_prices.csv').rename(columns={'player.name': 'Player'})
prices['Player'] = prices['Player'].str.strip()
print(f"Price data: {len(prices)} players")

# Load positions
try:
    pos_df = (pd.read_csv('euroleague_fantasy_analysis_complete.csv')
              [['player_name', 'estimated_position']]
              .drop_duplicates('player_name')
              .rename(columns={'player_name': 'Player', 'estimated_position': 'Position'}))
    pos_df['Position'] = pos_df['Position'].map(POSITION_MAP).fillna('?')
    print(f"Position data: {len(pos_df)} players")
except FileNotFoundError:
    pos_df = pd.DataFrame(columns=['Player', 'Position'])
    print("No position file found — position constraints disabled")

# Add target: next game FP per player
df['target'] = df.groupby('Player')['fantasy_points'].shift(-1)

# Ensure all feature cols exist
for col in FEATURE_COLS:
    if col not in df.columns:
        df[col] = 0.0

# Fill NaN in features
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

rounds = sorted(df['Round'].unique())
test_rounds = [r for r in rounds if r >= START_ROUND]
print(f"Backtesting {len(test_rounds)} rounds: {test_rounds[0]}–{test_rounds[-1]}\n")


# ── Helper: build LP-optimised team ─────────────────────────────────────────
def optimise_team(pool: pd.DataFrame, score_col: str, label: str) -> pd.DataFrame:
    """
    Pick the optimal 10-player team + captain from `pool`.
    `pool` must have columns: Player, Team, Position, price, + score_col.
    Returns the selected players as a DataFrame with a `is_captain` column.
    Returns None if infeasible.
    Automatically relaxes per-team cap when fewer than 4 teams are in the pool
    (playoff rounds with 2–3 teams).
    """
    pool = pool.dropna(subset=[score_col, 'price']).reset_index(drop=True)

    n_teams = pool['Team'].nunique()
    # In 2-team playoffs max 5/team; 3-team playoffs max 4/team; else standard 3/team
    per_team_cap = MAX_PER_TEAM if n_teams >= 4 else max(3, ROSTER // n_teams + 1)

    has_pos = all(
        pool[pool['Position'] == p].shape[0] >= req
        for p, req in POSITION_REQ.items()
    )

    def solve(use_positions: bool):
        prob = LpProblem(f"{label}_LP", LpMaximize)
        pv = {i: LpVariable(f"p_{i}", cat='Binary') for i in pool.index}
        cv = {i: LpVariable(f"c_{i}", cat='Binary') for i in pool.index}

        prob += lpSum(
            pv[i] * pool.loc[i, score_col] + cv[i] * pool.loc[i, score_col]
            for i in pool.index
        )
        prob += lpSum(pv[i] for i in pool.index) == ROSTER
        prob += lpSum(pv[i] * pool.loc[i, 'price'] for i in pool.index) <= BUDGET
        for team, grp in pool.groupby('Team'):
            prob += lpSum(pv[i] for i in grp.index) <= per_team_cap
        if use_positions:
            for pos, req in POSITION_REQ.items():
                prob += lpSum(pv[i] for i in pool[pool['Position'] == pos].index) == req
        prob += lpSum(cv[i] for i in pool.index) == 1
        for i in pool.index:
            prob += cv[i] <= pv[i]

        prob.solve(PULP_CBC_CMD(msg=0))
        if prob.status != 1:
            return None
        sel = [i for i in pool.index if pv[i].varValue == 1]
        cap = next((i for i in pool.index if cv[i].varValue == 1), None)
        out = pool.loc[sel].copy()
        out['is_captain'] = out.index == cap
        return out

    result = solve(has_pos)
    if result is None and has_pos:
        result = solve(False)
    return result


# ── Round-by-round walk-forward ───────────────────────────────────────────────
results = []

for r in test_rounds:
    # ── 1. Training set: rows from rounds < r with valid target ──────────────
    train = df[(df['Round'] < r) & df['target'].notna()].copy()
    if len(train) < 100:
        continue

    X_train = train[FEATURE_COLS]
    y_train = train['target']

    model = xgb.XGBRegressor(**BEST_PARAMS, random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train)

    # ── 2. Prediction features: each player's most recent row before round r ─
    pre_r = df[df['Round'] < r].copy()
    latest = pre_r.sort_values('Round').groupby('Player').tail(1).copy()
    latest['predicted_fp'] = model.predict(latest[FEATURE_COLS])

    # Attach prices and positions
    latest = latest.merge(prices, on='Player', how='inner')
    latest = latest.merge(pos_df, on='Player', how='left')
    latest['Position'] = latest['Position'].fillna('?')

    if len(latest) < ROSTER:
        continue

    # ── 3. Predicted team ────────────────────────────────────────────────────
    pred_team = optimise_team(latest, 'predicted_fp', f'pred_r{r}')
    if pred_team is None:
        continue

    # ── 4. Actual FP in round r ─────────────────────────────────────────────
    actual_r = (df[df['Round'] == r][['Player', 'fantasy_points', 'Team']]
                .rename(columns={'fantasy_points': 'actual_fp'}))

    # Score predicted team on actual FP
    pred_scored = pred_team.merge(actual_r[['Player', 'actual_fp']], on='Player', how='left')
    pred_scored['actual_fp'] = pred_scored['actual_fp'].fillna(0.0)
    # Captain = player we predicted to score highest on this team
    cap_idx = pred_scored['predicted_fp'].idxmax()
    pred_scored['is_captain'] = pred_scored.index == cap_idx
    pred_scored['effective_fp'] = np.where(
        pred_scored['is_captain'], pred_scored['actual_fp'] * 2, pred_scored['actual_fp']
    )
    pred_total = pred_scored['effective_fp'].sum()

    # ── 5. Naive baseline: top fp_last_3 team ────────────────────────────────
    naive_team = optimise_team(latest, 'fp_last_3', f'naive_r{r}')
    if naive_team is not None:
        naive_scored = naive_team.merge(actual_r[['Player', 'actual_fp']], on='Player', how='left')
        naive_scored['actual_fp'] = naive_scored['actual_fp'].fillna(0.0)
        naive_cap_idx = naive_scored['fp_last_3'].idxmax()
        naive_scored['effective_fp'] = np.where(
            naive_scored.index == naive_cap_idx,
            naive_scored['actual_fp'] * 2,
            naive_scored['actual_fp']
        )
        naive_total = naive_scored['effective_fp'].sum()
    else:
        naive_total = np.nan

    # ── 6. Oracle team (constrained): LP on actual FP ────────────────────────
    oracle_pool = (actual_r[actual_r['actual_fp'] > 0]
                   .merge(prices, on='Player', how='inner')
                   .merge(pos_df, on='Player', how='left'))
    oracle_pool['Position'] = oracle_pool['Position'].fillna('?')

    oracle_team = optimise_team(oracle_pool, 'actual_fp', f'oracle_r{r}')
    if oracle_team is not None:
        oracle_scored = oracle_team.copy()
        oracle_cap_idx = oracle_scored['actual_fp'].idxmax()
        oracle_scored['effective_fp'] = np.where(
            oracle_scored.index == oracle_cap_idx,
            oracle_scored['actual_fp'] * 2,
            oracle_scored['actual_fp']
        )
        oracle_total = oracle_scored['effective_fp'].sum()
    else:
        oracle_total = np.nan

    # ── 7. Oracle unconstrained: top-10 by actual FP ─────────────────────────
    top10_actual = actual_r.nlargest(10, 'actual_fp')
    oracle_unconstrained = top10_actual['actual_fp'].sum() + top10_actual['actual_fp'].max()

    # ── 8. How many of our predicted team appeared in oracle team? ───────────
    oracle_players = set(oracle_team['Player']) if oracle_team is not None else set()
    pred_players   = set(pred_team['Player'])
    overlap = len(pred_players & oracle_players)

    results.append({
        'round':                   r,
        'predicted_team_fp':       round(pred_total, 1),
        'naive_baseline_fp':       round(naive_total, 1) if not np.isnan(naive_total) else None,
        'oracle_constrained_fp':   round(oracle_total, 1) if not np.isnan(oracle_total) else None,
        'oracle_unconstrained_fp': round(oracle_unconstrained, 1),
        'oracle_overlap':          overlap,
        'captain':                 pred_scored.loc[cap_idx, 'Player'] if cap_idx is not None else '',
        'captain_actual_fp':       round(pred_scored.loc[cap_idx, 'actual_fp'], 1) if cap_idx is not None else 0,
    })

    pct = (pred_total / oracle_total * 100) if oracle_total and not np.isnan(oracle_total) else 0
    print(f"Round {r:>2}  ML={pred_total:>6.1f}  Naive={naive_total:>6.1f}  "
          f"Oracle={oracle_total:>6.1f}  ({pct:.0f}%)  "
          f"Overlap={overlap}/10  Captain: {results[-1]['captain']}")


# ── Summary ───────────────────────────────────────────────────────────────────
res_df = pd.DataFrame(results)
res_df.to_csv('backtest_results.csv', index=False)

print("\n" + "=" * 65)
print("BACKTEST SUMMARY")
print("=" * 65)

valid = res_df.dropna(subset=['oracle_constrained_fp'])
ml_mean    = valid['predicted_team_fp'].mean()
naive_mean = valid['naive_baseline_fp'].mean()
oracle_mean = valid['oracle_constrained_fp'].mean()
uncon_mean  = valid['oracle_unconstrained_fp'].mean()

print(f"\nRounds evaluated: {len(valid)}")
print(f"\n{'Metric':<30} {'Avg FP/round':>12}")
print("-" * 45)
print(f"{'ML predicted team':<30} {ml_mean:>12.1f}")
print(f"{'Naive baseline (fp_last_3)':<30} {naive_mean:>12.1f}")
print(f"{'Oracle (constrained)':<30} {oracle_mean:>12.1f}")
print(f"{'Oracle (unconstrained top-10)':<30} {uncon_mean:>12.1f}")

print(f"\nML vs Naive:   {ml_mean - naive_mean:+.1f} pts/round")
print(f"ML vs Oracle:  {ml_mean - oracle_mean:+.1f} pts/round  "
      f"({ml_mean / oracle_mean * 100:.1f}% of oracle)")

print(f"\nAvg oracle overlap (players in common): "
      f"{valid['oracle_overlap'].mean():.1f} / 10")

print(f"\nBest rounds (ML):")
print(valid.nlargest(5, 'predicted_team_fp')
      [['round', 'predicted_team_fp', 'oracle_constrained_fp', 'oracle_overlap', 'captain']]
      .to_string(index=False))

print(f"\nWorst rounds (ML):")
print(valid.nsmallest(5, 'predicted_team_fp')
      [['round', 'predicted_team_fp', 'oracle_constrained_fp', 'oracle_overlap', 'captain']]
      .to_string(index=False))

print(f"\nRounds where ML beat naive: "
      f"{(valid['predicted_team_fp'] > valid['naive_baseline_fp']).sum()} / {len(valid)}")

print("\n✓ Saved: backtest_results.csv")
print("=" * 65)