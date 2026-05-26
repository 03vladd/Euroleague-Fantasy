"""
backtest_subs.py — Quantify the value of mid-round lineup substitutions.

For every backtest round, rebuilds the ML team and then computes three scores:

  1. Original       — lineup as chosen by the LP (no hindsight)
  2. Oracle sub     — after Day-1 results are known, reassign starter/bench/captain
                      among Day-2 players using perfect hindsight of Day-2 FP
  3. DNP-aware sub  — same reassignment, but only move players who had 0 minutes
                      (announced DNPs) from starter → bench; re-assign captain
                      using predicted FP for the remaining Day-2 players
                      (models reading injury news before Day-2 tipoff)

Output: backtest_sub_results.csv + summary
"""

import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD

from euroleague_api.schedule import Schedule

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
START_ROUND   = 6
BUDGET        = 100
ROSTER        = 10
MAX_PER_TEAM  = 6
STARTERS      = 6
POSITION_REQ  = {'G': 4, 'F': 4, 'C': 2}
POSITION_MAP  = {'Guard': 'G', 'Forward': 'F', 'Center': 'C'}
SEASON        = 2025

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
print("MID-ROUND SUBSTITUTION BACKTEST")
print("=" * 65)

df = pd.read_csv('data/processed/player_games_with_features.csv')
df = df[df['Minutes'] >= 5].copy()
df = df.sort_values(['Player', 'Round']).reset_index(drop=True)
df['target'] = df.groupby('Player')['fantasy_points'].shift(-1)
for col in FEATURE_COLS:
    if col not in df.columns:
        df[col] = 0.0
df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

with open('models/best_params.json') as f:
    BEST_PARAMS = json.load(f)

prices = pd.read_csv('euroleague_prices.csv').rename(columns={'player.name': 'Player'})
prices['Player'] = prices['Player'].str.strip()
pos_df = (pd.read_csv('euroleague_fantasy_analysis_complete.csv')
          [['player_name', 'estimated_position']]
          .drop_duplicates('player_name')
          .rename(columns={'player_name': 'Player', 'estimated_position': 'Position'}))
pos_df['Position'] = pos_df['Position'].map(POSITION_MAP).fillna('?')

# ── Schedule: build team_code → game_date map for every round ────────────────
print("Fetching EuroLeague schedule…")
sched_raw = Schedule('E').get_schedule(SEASON)

def parse_dt(row):
    try:
        return datetime.strptime(f"{row['date']} {row['startime']}", "%b %d, %Y %H:%M")
    except Exception:
        return None

sched_raw['gametime'] = sched_raw.apply(parse_dt, axis=1)
sched_raw = sched_raw.dropna(subset=['gametime'])


def team_day_map(gameday: int) -> dict[str, object]:
    """Return {team_code: game_date} for every team in the round."""
    r = sched_raw[sched_raw['gameday'] == gameday]
    mapping: dict[str, object] = {}
    for _, row in r.iterrows():
        d = row['gametime'].date()
        for code in (row['homecode'], row['awaycode']):
            if code not in mapping or d < mapping[code]:
                mapping[code] = d
    return mapping


# ── LP solver ─────────────────────────────────────────────────────────────────
def optimise_team(pool: pd.DataFrame, score_col: str) -> pd.DataFrame | None:
    pool = pool.dropna(subset=[score_col, 'price']).reset_index(drop=True)
    n_teams = pool['Team'].nunique()
    per_team_cap = MAX_PER_TEAM if n_teams >= 4 else max(3, ROSTER // n_teams + 1)
    has_pos = all(pool[pool['Position'] == p].shape[0] >= r
                  for p, r in POSITION_REQ.items())

    def solve(use_pos):
        prob = LpProblem('LP', LpMaximize)
        pv = {i: LpVariable(f'p_{i}', cat='Binary') for i in pool.index}
        sv = {i: LpVariable(f's_{i}', cat='Binary') for i in pool.index}
        cv = {i: LpVariable(f'c_{i}', cat='Binary') for i in pool.index}
        sc = pool[score_col]
        prob += lpSum(sc[i]*0.5*pv[i] + sc[i]*0.5*sv[i] + sc[i]*cv[i] for i in pool.index)
        prob += lpSum(pv[i] for i in pool.index) == ROSTER
        prob += lpSum(sv[i] for i in pool.index) == STARTERS
        prob += lpSum(pv[i]*pool.loc[i, 'price'] for i in pool.index) <= BUDGET
        for team, grp in pool.groupby('Team'):
            prob += lpSum(pv[i] for i in grp.index) <= per_team_cap
        if use_pos:
            for pos, req in POSITION_REQ.items():
                prob += lpSum(pv[i] for i in pool[pool['Position'] == pos].index) == req
        prob += lpSum(cv[i] for i in pool.index) == 1
        for i in pool.index:
            prob += sv[i] <= pv[i]
            prob += cv[i] <= sv[i]
        prob.solve(PULP_CBC_CMD(msg=0))
        if prob.status != 1:
            return None
        sel    = [i for i in pool.index if pv[i].varValue == 1]
        cap    = next((i for i in pool.index if cv[i].varValue == 1), None)
        starts = {i for i in pool.index if sv[i].varValue == 1}
        out = pool.loc[sel].copy()
        out['is_captain'] = out.index == cap
        out['is_starter'] = out.index.map(lambda i: i in starts)
        return out

    result = solve(has_pos)
    return result if result is not None else solve(False)


# ── Score helper ──────────────────────────────────────────────────────────────
def score_with_roles(team: pd.DataFrame, actual_col: str,
                     starter_set: set, captain_idx) -> float:
    """Score a team given explicit starter/captain assignments."""
    total = 0.0
    for i, row in team.iterrows():
        fp = row[actual_col]
        if i == captain_idx:
            total += fp * 2
        elif i in starter_set:
            total += fp
        else:
            total += fp * 0.5
    return total


def optimal_assignment(team: pd.DataFrame, score_col: str,
                        locked_starters: set, locked_bench: set,
                        free_count: int) -> tuple[set, object]:
    """
    Assign starter slots to the `free_count` best free players by `score_col`.
    Returns (new_starter_set, captain_idx).
    captain = highest score_col across all starters (locked + free).
    """
    free = team[~team.index.isin(locked_starters | locked_bench)]
    free_starters = set(free.nlargest(free_count, score_col).index)
    all_starters  = locked_starters | free_starters
    captain_idx   = team.loc[list(all_starters), score_col].idxmax()
    return all_starters, captain_idx


# ── Walk-forward loop ─────────────────────────────────────────────────────────
rounds   = sorted(df['Round'].unique())
test_rds = [r for r in rounds if r >= START_ROUND]
results  = []

for r in test_rds:
    # ── 1. Train ─────────────────────────────────────────────────────────────
    train = df[(df['Round'] < r) & df['target'].notna()]
    if len(train) < 100:
        continue
    model = xgb.XGBRegressor(**BEST_PARAMS, random_state=42, n_jobs=-1, verbosity=0)
    model.fit(train[FEATURE_COLS], train['target'])

    # ── 2. Prediction pool ────────────────────────────────────────────────────
    pre_r  = df[df['Round'] < r].copy()
    latest = pre_r.sort_values('Round').groupby('Player').tail(1).copy()
    latest['predicted_fp'] = model.predict(latest[FEATURE_COLS])
    latest = (latest.merge(prices, on='Player', how='inner')
                     .merge(pos_df, on='Player', how='left'))
    latest['Position'] = latest['Position'].fillna('?')
    if len(latest) < ROSTER:
        continue

    # ── 3. ML team ───────────────────────────────────────────────────────────
    pred_team = optimise_team(latest, 'predicted_fp')
    if pred_team is None:
        continue

    # ── 4. Actual FP for round r ──────────────────────────────────────────────
    actual_r = (df[df['Round'] == r][['Player', 'fantasy_points', 'Minutes', 'Team']]
                .rename(columns={'fantasy_points': 'actual_fp', 'Minutes': 'actual_minutes'}))

    t = pred_team.merge(actual_r[['Player', 'actual_fp', 'actual_minutes']],
                        on='Player', how='left')
    t['actual_fp']      = t['actual_fp'].fillna(0.0)
    t['actual_minutes'] = t['actual_minutes'].fillna(0.0)

    # ── 5. Schedule: split by game day ───────────────────────────────────────
    tdm    = team_day_map(r)
    t['game_day'] = t['Team'].map(tdm)

    days_in_round = sorted(t['game_day'].dropna().unique())
    multi_day = len(days_in_round) >= 2

    # ── 6. Original score (LP lineup) ────────────────────────────────────────
    orig_starters = set(t[t['is_starter']].index)
    orig_captain  = t[t['is_captain']].index[0] if t['is_captain'].any() else next(iter(orig_starters))
    score_orig    = score_with_roles(t, 'actual_fp', orig_starters, orig_captain)

    # ── 7. Oracle sub & DNP-aware sub ────────────────────────────────────────
    score_oracle_sub = score_orig  # default: no sub possible (single-day round)
    score_dnp_sub    = score_orig
    n_days_in_round  = len(days_in_round)
    n_day1_players   = 0
    n_dnps_subbed    = 0

    if multi_day:
        day1 = days_in_round[0]
        t_d1 = t[t['game_day'] == day1]   # played on Day 1
        t_d2 = t[t['game_day'] >  day1]   # play on Day 2+ (haven't played yet)

        n_day1_players = len(t_d1)
        d1_starters    = orig_starters & set(t_d1.index)
        d1_bench       = set(t_d1.index) - d1_starters
        d2_starter_slots = STARTERS - len(d1_starters)

        # ── Oracle sub: unconstrained retroactive reassignment ────────────────
        # Any player (D1 or D2, starter or bench) can be reassigned.
        # Pick the 6 highest actual-FP players as starters; best one is captain.
        top6_oracle   = set(t.nlargest(STARTERS, 'actual_fp').index)
        d2_cap_oracle = t.loc[list(top6_oracle), 'actual_fp'].idxmax()
        score_oracle_sub = score_with_roles(t, 'actual_fp', top6_oracle, d2_cap_oracle)

        # ── DNP-aware sub: only move 0-minute players to bench ───────────────
        # Mark Day-2 DNPs (0 minutes AND not in actual game data at all)
        d2_dnp = set(t[(t['game_day'] > day1) & (t['actual_minutes'] == 0.0)].index)
        n_dnps_subbed = len(d2_dnp & orig_starters)

        # Forced bench: Day-2 players with 0 minutes (known DNPs before they play)
        # Reassign captain among remaining Day-2 starters using predicted FP
        locked_bench_dnp = d1_bench | d2_dnp
        d2_active_starters = (orig_starters - d2_dnp) & set(t_d2.index)
        # Available free slots: if DNPs freed up starter spots, fill from bench
        free_slots = max(0, d2_starter_slots - len(d2_active_starters))
        d2_bench_avail = (set(t_d2.index) - d2_dnp) - d2_active_starters
        # Promote best bench players (by predicted_fp) to fill freed starter slots
        promoted = set(
            t.loc[list(d2_bench_avail)].nlargest(free_slots, 'predicted_fp').index
        ) if free_slots > 0 and d2_bench_avail else set()
        d2_stars_dnp = d1_starters | d2_active_starters | promoted
        # Captain = best predicted_fp among all starters (we still don't know Day-2 actuals)
        d2_cap_dnp = t.loc[list(d2_stars_dnp), 'predicted_fp'].idxmax()
        score_dnp_sub = score_with_roles(t, 'actual_fp', d2_stars_dnp, d2_cap_dnp)

    results.append({
        'round':            r,
        'n_game_days':      n_days_in_round,
        'n_day1_players':   n_day1_players,
        'score_orig':       round(score_orig, 1),
        'score_oracle_sub': round(score_oracle_sub, 1),
        'score_dnp_sub':    round(score_dnp_sub, 1),
        'oracle_gain':      round(score_oracle_sub - score_orig, 1),
        'dnp_gain':         round(score_dnp_sub - score_orig, 1),
        'n_dnps_subbed':    n_dnps_subbed,
    })

    print(f"Round {r:>2}  days={n_days_in_round}  "
          f"orig={score_orig:>6.1f}  "
          f"oracle_sub={score_oracle_sub:>6.1f} (+{score_oracle_sub-score_orig:.1f})  "
          f"dnp_sub={score_dnp_sub:>6.1f} (+{score_dnp_sub-score_orig:.1f})  "
          f"dnps={n_dnps_subbed}")


# ── Summary ───────────────────────────────────────────────────────────────────
res = pd.DataFrame(results)
res.to_csv('backtest_sub_results.csv', index=False)

print("\n" + "=" * 65)
print("SUBSTITUTION BACKTEST SUMMARY")
print("=" * 65)

multi = res[res['n_game_days'] >= 2]
single = res[res['n_game_days'] == 1]

print(f"\nRounds evaluated: {len(res)}  "
      f"(multi-day: {len(multi)}, single-day: {len(single)})")

print(f"\n{'Metric':<35} {'Avg FP/round':>12}")
print("-" * 50)
print(f"{'Original ML lineup':<35} {res['score_orig'].mean():>12.1f}")
print(f"{'Oracle sub (full retroactive reassign)':<35} {res['score_oracle_sub'].mean():>12.1f}")
print(f"{'DNP-aware sub (injury news only)':<35} {res['score_dnp_sub'].mean():>12.1f}")

print(f"\nOracle sub gain (all rounds):       +{res['oracle_gain'].mean():.1f} pts/round")
print(f"Oracle sub gain (multi-day rounds): +{multi['oracle_gain'].mean():.1f} pts/round")
print(f"DNP-aware sub gain (all rounds):    +{res['dnp_gain'].mean():.1f} pts/round")

print(f"\nRounds where oracle sub helps: "
      f"{(res['oracle_gain'] > 0).sum()} / {len(res)}")
print(f"Rounds where DNP sub helps:    "
      f"{(res['dnp_gain'] > 0).sum()} / {len(res)}")

print(f"\nBest oracle-sub gains:")
print(res.nlargest(5, 'oracle_gain')
      [['round', 'n_game_days', 'score_orig', 'score_oracle_sub', 'oracle_gain']]
      .to_string(index=False))

print(f"\nBest DNP-sub gains:")
print(res.nlargest(5, 'dnp_gain')
      [['round', 'n_game_days', 'n_dnps_subbed', 'score_orig', 'score_dnp_sub', 'dnp_gain']]
      .to_string(index=False))

print(f"\nAvg DNPs subbed per round (multi-day): {multi['n_dnps_subbed'].mean():.2f}")
print("\n✓ Saved: backtest_sub_results.csv")
print("=" * 65)