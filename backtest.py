"""
backtest.py — Round-by-round backtest of ML fantasy team predictions.

Fsor each round R from START_ROUND to the last round:
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


# ── Coach scoring formula ─────────────────────────────────────────────────────
def _coach_fp(margin: int, won: bool) -> int:
    if won:
        return 10 if margin <= 10 else (20 if margin <= 20 else 25)
    else:
        return -5 if margin <= 10 else (-10 if margin <= 20 else -20)

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
START_ROUND = 6      # need enough history for rolling features to be meaningful
END_ROUND   = 38     # last regular-season round
BUDGET      = 100
ROSTER      = 10
MAX_PER_TEAM = 6
POSITION_REQ = {'G': 4, 'F': 4, 'C': 2}
POSITION_MAP = {'Guard': 'G', 'Forward': 'F', 'Center': 'C'}

import pickle
with open('models/feature_columns.pkl', 'rb') as _f:
    FEATURE_COLS = pickle.load(_f)

# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("EUROLEAGUE FANTASY BACKTEST")
print("=" * 65)

# Pre-compute coach FP per team per round from boxscore team totals
_raw = pd.read_csv('data/raw/all_player_game_stats_2025.csv')
_el  = _raw[(_raw['Competition'] == 'Euroleague') & (_raw['Player'] == 'Total')].copy()
_home = _el[_el['Home'] == 1][['Round', 'Gamecode', 'Team', 'Points']].rename(
    columns={'Team': 'home_team', 'Points': 'home_pts'})
_away = _el[_el['Home'] == 0][['Round', 'Gamecode', 'Team', 'Points']].rename(
    columns={'Team': 'away_team', 'Points': 'away_pts'})
_games = _home.merge(_away, on=['Round', 'Gamecode'])
_games['margin']   = (_games['home_pts'] - _games['away_pts']).abs()
_games['home_win'] = _games['home_pts'] > _games['away_pts']
_coach_home = _games[['Round', 'home_team', 'home_win', 'margin']].copy()
_coach_home['team'] = _coach_home['home_team']
_coach_home['won']  = _coach_home['home_win']
_coach_away = _games[['Round', 'away_team', 'home_win', 'margin']].copy()
_coach_away['team'] = _coach_away['away_team']
_coach_away['won']  = ~_coach_away['home_win']
COACH_FP = pd.concat([
    _coach_home[['Round', 'team', 'won', 'margin']],
    _coach_away[['Round', 'team', 'won', 'margin']],
], ignore_index=True)
COACH_FP['coach_fp'] = COACH_FP.apply(
    lambda r: _coach_fp(int(r['margin']), bool(r['won'])), axis=1)
# Dict: round → {team: coach_fp}
COACH_FP_DICT: dict[int, dict[str, int]] = {}
for rnd, grp in COACH_FP.groupby('Round'):
    COACH_FP_DICT[int(rnd)] = dict(zip(grp['team'], grp['coach_fp']))

print(f"Coach data: {len(COACH_FP_DICT)} rounds loaded")

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

# Load prices — prefer the processed file (built by prepare_data.py) which has
# full-season coverage; fall back to euroleague_prices.csv if only that exists
_price_src = 'data/processed/all_players_with_prices.csv'
if not Path(_price_src).exists():
    _price_src = 'euroleague_prices.csv'
prices = (pd.read_csv(_price_src)[['player.name', 'price']]
          .rename(columns={'player.name': 'Player'})
          .drop_duplicates('Player'))
prices['Player'] = prices['Player'].str.strip()
print(f"Price data: {len(prices)} players  (from {_price_src})")

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
test_rounds = [r for r in rounds if START_ROUND <= r <= END_ROUND]
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

    STARTERS = 6  # 6 players at 100%, 4 bench at 50%

    def solve(use_positions: bool):
        prob = LpProblem(f"{label}_LP", LpMaximize)
        pv = {i: LpVariable(f"p_{i}", cat='Binary') for i in pool.index}
        sv = {i: LpVariable(f"s_{i}", cat='Binary') for i in pool.index}
        cv = {i: LpVariable(f"c_{i}", cat='Binary') for i in pool.index}

        sc = pool[score_col]
        prob += lpSum(
            sc[i] * 0.5 * pv[i] +  # bench baseline
            sc[i] * 0.5 * sv[i] +  # starter bonus
            sc[i] * cv[i]           # captain bonus
            for i in pool.index
        )
        prob += lpSum(pv[i] for i in pool.index) == ROSTER
        prob += lpSum(sv[i] for i in pool.index) == STARTERS
        prob += lpSum(pv[i] * pool.loc[i, 'price'] for i in pool.index) <= BUDGET
        for team, grp in pool.groupby('Team'):
            prob += lpSum(pv[i] for i in grp.index) <= per_team_cap
        if use_positions:
            for pos, req in POSITION_REQ.items():
                prob += lpSum(pv[i] for i in pool[pool['Position'] == pos].index) == req
        prob += lpSum(cv[i] for i in pool.index) == 1
        for i in pool.index:
            prob += sv[i] <= pv[i]
            prob += cv[i] <= sv[i]  # captain must be a starter

        prob.solve(PULP_CBC_CMD(msg=0))
        if prob.status != 1:
            return None
        sel = [i for i in pool.index if pv[i].varValue == 1]
        cap = next((i for i in pool.index if cv[i].varValue == 1), None)
        starters = {i for i in pool.index if sv[i].varValue == 1}
        out = pool.loc[sel].copy()
        out['is_captain'] = out.index == cap
        out['is_starter'] = out.index.map(lambda i: i in starters)
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
    def score_team(team_df, actual_fp_col, pred_col):
        """Score a team using 6-starter / 4-bench / 1-captain rules."""
        t = team_df.copy()
        t['actual_fp'] = t['actual_fp'] if 'actual_fp' in t.columns else 0.0
        # Assign starter slots to top-6 by predicted score (optimal ordering)
        top6 = t.nlargest(6, pred_col).index
        t['is_starter'] = t.index.map(lambda i: i in set(top6))
        cap_idx = t.loc[top6, pred_col].idxmax()
        t['is_captain'] = t.index == cap_idx
        t['effective_fp'] = np.where(
            t['is_captain'], t[actual_fp_col] * 2,
            np.where(t['is_starter'], t[actual_fp_col], t[actual_fp_col] * 0.5)
        )
        return t['effective_fp'].sum()

    pred_scored = pred_team.merge(actual_r[['Player', 'actual_fp']], on='Player', how='left')
    pred_scored['actual_fp'] = pred_scored['actual_fp'].fillna(0.0)
    pred_total = score_team(pred_scored, 'actual_fp', 'predicted_fp')
    # Identify captain for reporting (top predicted among starters)
    top6_pred = pred_scored.nlargest(6, 'predicted_fp')
    cap_idx = top6_pred['predicted_fp'].idxmax()
    cap_player = pred_scored.loc[cap_idx, 'Player']
    cap_actual = pred_scored.loc[cap_idx, 'actual_fp']

    # ── 5. Naive baseline: top fp_last_3 team ────────────────────────────────
    naive_team = optimise_team(latest, 'fp_last_3', f'naive_r{r}')
    if naive_team is not None:
        naive_scored = naive_team.merge(actual_r[['Player', 'actual_fp']], on='Player', how='left')
        naive_scored['actual_fp'] = naive_scored['actual_fp'].fillna(0.0)
        naive_total = score_team(naive_scored, 'actual_fp', 'fp_last_3')
    else:
        naive_total = np.nan

    # ── 6. Oracle team (constrained): LP on actual FP ────────────────────────
    oracle_pool = (actual_r[actual_r['actual_fp'] > 0]
                   .merge(prices, on='Player', how='inner')
                   .merge(pos_df, on='Player', how='left'))
    oracle_pool['Position'] = oracle_pool['Position'].fillna('?')

    oracle_team = optimise_team(oracle_pool, 'actual_fp', f'oracle_r{r}')
    if oracle_team is not None:
        oracle_total = score_team(oracle_team, 'actual_fp', 'actual_fp')
    else:
        oracle_total = np.nan

    # ── 7. Oracle unconstrained: top-10 by actual FP ─────────────────────────
    top10_actual = actual_r.nlargest(10, 'actual_fp')
    top6_unc = top10_actual.nlargest(6, 'actual_fp')
    oracle_unconstrained = (
        top6_unc['actual_fp'].max() * 2 +          # captain
        top6_unc['actual_fp'].sum() - top6_unc['actual_fp'].max() +  # other 5 starters
        top10_actual.nsmallest(4, 'actual_fp')['actual_fp'].sum() * 0.5  # bench
    )

    # ── 8. How many of our predicted team appeared in oracle team? ───────────
    oracle_players = set(oracle_team['Player']) if oracle_team is not None else set()
    pred_players   = set(pred_team['Player'])
    overlap = len(pred_players & oracle_players)

    # ── 9. Coach scoring ─────────────────────────────────────────────────────
    round_coach = COACH_FP_DICT.get(r, {})
    # Oracle coach: best team this round
    oracle_coach_fp = max(round_coach.values()) if round_coach else 0
    # Naive coach: pick team with best cumulative coach FP so far
    prior_coach = COACH_FP[COACH_FP['Round'] < r].groupby('team')['coach_fp'].sum()
    naive_coach_team = prior_coach.idxmax() if not prior_coach.empty else None
    naive_coach_fp = round_coach.get(naive_coach_team, 0) if naive_coach_team else 0
    # ML-team coach: pick the team with most players in our predicted team
    team_counts = pred_scored['Team'].value_counts()
    ml_coach_team = team_counts.index[0] if not team_counts.empty else None
    ml_coach_fp = round_coach.get(ml_coach_team, 0) if ml_coach_team else 0

    results.append({
        'round':                   r,
        'predicted_team_fp':       round(pred_total, 1),
        'naive_baseline_fp':       round(naive_total, 1) if not np.isnan(naive_total) else None,
        'oracle_constrained_fp':   round(oracle_total, 1) if not np.isnan(oracle_total) else None,
        'oracle_unconstrained_fp': round(oracle_unconstrained, 1),
        'oracle_overlap':          overlap,
        'captain':                 cap_player,
        'captain_actual_fp':       round(cap_actual, 1),
        'oracle_coach_fp':         oracle_coach_fp,
        'naive_coach_fp':          naive_coach_fp,
        'ml_team_coach_fp':        ml_coach_fp,
        'oracle_coach_team':       max(round_coach, key=round_coach.get) if round_coach else None,
        'naive_coach_team':        naive_coach_team,
    })

    pct = (pred_total / oracle_total * 100) if oracle_total and not np.isnan(oracle_total) else 0
    print(f"Round {r:>2}  ML={pred_total:>6.1f}  Naive={naive_total:>6.1f}  "
          f"Oracle={oracle_total:>6.1f}  ({pct:.0f}%)  "
          f"Coach: oracle={oracle_coach_fp:+d} naive={naive_coach_fp:+d}  "
          f"Captain: {results[-1]['captain']}")


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

# ── Coach scoring summary ─────────────────────────────────────────────────────
oracle_coach_mean = valid['oracle_coach_fp'].mean()
naive_coach_mean  = valid['naive_coach_fp'].mean()
ml_coach_mean     = valid['ml_team_coach_fp'].mean()

print(f"\n{'─'*45}")
print(f"COACH SCORING (separate from player FP):")
print(f"{'─'*45}")
print(f"\n{'Metric':<35} {'Avg FP/round':>12}")
print("-" * 50)
print(f"{'Oracle coach (best pick each round)':<35} {oracle_coach_mean:>+12.1f}")
print(f"{'Naive coach (best cumul. record)':<35} {naive_coach_mean:>+12.1f}")
print(f"{'ML-team coach (most players team)':<35} {ml_coach_mean:>+12.1f}")
print(f"\nML players + oracle coach:  {ml_mean + oracle_coach_mean:.1f} pts/round")
print(f"ML players + naive coach:   {ml_mean + naive_coach_mean:.1f} pts/round")
print(f"\nOracle coach most common picks:")
top_oracle_coaches = valid['oracle_coach_team'].value_counts().head(5)
for team, cnt in top_oracle_coaches.items():
    print(f"  {team}: {cnt} rounds")

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