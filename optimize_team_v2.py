"""
Optimize fantasy team using ML predictions.
Enforces 4G/4F/2C position constraints and picks an optimal captain (2× points).

Usage:
    python3 optimize_team_v2.py                 # no schedule info
    python3 optimize_team_v2.py --round 5       # show game times for round 5
"""

import argparse
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD

parser = argparse.ArgumentParser()
parser.add_argument('--round',  type=int, default=None, help='Round number for schedule display')
parser.add_argument('--season', type=int, default=2025)
args = parser.parse_args()

print("=" * 60)
print("EUROLEAGUE FANTASY OPTIMIZER V2 (ML-POWERED)")
print("=" * 60)

# ── Load predictions ──────────────────────────────────────────
df = pd.read_csv('data/processed/next_round_predictions.csv')
df = df[df['games_played_max'] >= 3].copy()  # relaxed from 5 to not cut early-season hot streaks
df = df.reset_index(drop=True)

# ── Attach position data ──────────────────────────────────────
# euroleague_fantasy_analysis_complete.csv carries estimated_position (Guard/Forward/Center)
POSITION_MAP = {'Guard': 'G', 'Forward': 'F', 'Center': 'C'}
try:
    analysis = pd.read_csv('euroleague_fantasy_analysis_complete.csv')
    pos_df = (analysis[['player_name', 'estimated_position']]
              .drop_duplicates('player_name')
              .rename(columns={'player_name': 'Player', 'estimated_position': 'Position'}))
    pos_df['Position'] = pos_df['Position'].map(POSITION_MAP).fillna('?')
    df = df.merge(pos_df, on='Player', how='left')
    df['Position'] = df['Position'].fillna('?')
    matched = df['Position'].ne('?').sum()
    print(f"\nPosition data: {matched}/{len(df)} players matched")
except FileNotFoundError:
    df['Position'] = '?'
    print("\nWarning: euroleague_fantasy_analysis_complete.csv not found — no position constraints")

print(f"Available players: {len(df)}")

# ── Config ────────────────────────────────────────────────────
BUDGET     = 100
ROSTER     = 10
MAX_PER_TEAM   = 6
REQ = {'G': 4, 'F': 4, 'C': 2}

has_positions = all(
    df[df['Position'] == pos].shape[0] >= req
    for pos, req in REQ.items()
)

print(f"\nConstraints:")
print(f"  Budget: {BUDGET} credits | Roster: {ROSTER} | Max/team: {MAX_PER_TEAM}")
if has_positions:
    print(f"  Positions: {REQ['G']}G / {REQ['F']}F / {REQ['C']}C")
else:
    print("  Positions: skipped (not enough position data)")

# ── Build LP problem ─────────────────────────────────────────
# Scoring: 6 starters (100% FP) + 4 bench (50% FP) + captain (extra 1× on starter)
# starter_vars[i]=1 means player i is in the starting 6 (gets full points)
# bench contribution = 0.5 × predicted_fp × (player_vars[i] - starter_vars[i])
# Objective rewritten: 0.5×player + 0.5×starter + 1×captain
STARTERS = 6

def build_prob(name, use_positions):
    prob = LpProblem(name, LpMaximize)
    pv = {i: LpVariable(f"p_{i}", cat='Binary') for i in df.index}
    sv = {i: LpVariable(f"s_{i}", cat='Binary') for i in df.index}
    cv = {i: LpVariable(f"c_{i}", cat='Binary') for i in df.index}

    fp = df['predicted_fp']
    prob += lpSum(
        fp[i] * 0.5 * pv[i] +   # bench baseline (50%)
        fp[i] * 0.5 * sv[i] +   # starter bonus (total 100%)
        fp[i] * cv[i]            # captain bonus (total 200% for starter-captain)
        for i in df.index
    )
    prob += lpSum(pv[i] for i in df.index) == ROSTER
    prob += lpSum(sv[i] for i in df.index) == STARTERS
    prob += lpSum(pv[i] * df.loc[i, 'price'] for i in df.index) <= BUDGET
    for team, grp in df.groupby('Team'):
        prob += lpSum(pv[i] for i in grp.index) <= MAX_PER_TEAM
    if use_positions:
        for pos, req in REQ.items():
            pos_idx = df[df['Position'] == pos].index
            prob += lpSum(pv[i] for i in pos_idx) == req
    prob += lpSum(cv[i] for i in df.index) == 1
    for i in df.index:
        prob += sv[i] <= pv[i]   # must be selected to start
        prob += cv[i] <= sv[i]   # captain must be a starter
    return prob, pv, sv, cv

print("\nOptimizing...")
prob, player_vars, starter_vars, captain_vars = build_prob("Fantasy_Team_ML", has_positions)
prob.solve(PULP_CBC_CMD(msg=0))

if prob.status != 1:
    print(f"\nWARNING: Infeasible (status {prob.status})")
    if has_positions:
        print("Retrying without position constraints...")
        prob, player_vars, starter_vars, captain_vars = build_prob("Fantasy_Team_ML_NoPos", False)
        prob.solve(PULP_CBC_CMD(msg=0))
        if prob.status != 1:
            print("Still infeasible — check budget or player pool.")
        else:
            print("✓ Solution found (without position constraints)")
else:
    print("✓ Optimal solution found")

# ── Extract results ───────────────────────────────────────────
selected = [i for i in df.index if player_vars[i].varValue == 1]
captain  = next((i for i in df.index if captain_vars[i].varValue == 1), None)

team_df = df.loc[selected].copy()
team_df['is_captain'] = team_df.index == captain
team_df['is_starter'] = [starter_vars[i].varValue == 1 for i in team_df.index]
team_df['effective_fp'] = np.where(
    team_df['is_captain'],
    team_df['predicted_fp'] * 2,          # captain: 2×
    np.where(
        team_df['is_starter'],
        team_df['predicted_fp'],           # starter: 1×
        team_df['predicted_fp'] * 0.5     # bench: 0.5×
    )
)
team_df = team_df.sort_values('effective_fp', ascending=False)

print("\n" + "=" * 60)
print("OPTIMAL TEAM")
print("=" * 60)
print(f"\n{'★':<2} {'S':<2} {'Player':<25} {'Pos':<4} {'Team':<5} {'€':>5} {'L3':>6} {'Eff':>6} {'Avg':>6}")
print("-" * 76)

total_cost = total_fp = total_avg = 0
for _, p in team_df.iterrows():
    cap   = "★" if p['is_captain'] else " "
    start = "S" if p['is_starter'] else "B"
    pos   = p.get('Position', '?')
    print(f"{cap:<2} {start:<2} {p['Player']:<25} {pos:<4} {p['Team']:<5} "
          f"{p['price']:>5.1f} {p['fp_last_3']:>6.1f} "
          f"{p['effective_fp']:>6.1f} {p['fantasy_points_mean']:>6.1f}")
    total_cost += p['price']
    total_fp   += p['effective_fp']
    total_avg  += p['fantasy_points_mean']

print("-" * 76)
print(f"{'TOTAL':<35} {total_cost:>5.1f} {'':>6} {total_fp:>6.1f} {total_avg:>6.1f}")

if captain is not None:
    cap_row = df.loc[captain]
    print(f"\n★ Captain: {cap_row['Player']}  "
          f"({cap_row['predicted_fp']:.1f} → {cap_row['predicted_fp']*2:.1f} pts)")

print(f"  Budget remaining: {BUDGET - total_cost:.1f} credits")
print(f"  Expected total (6 starters 100%, 4 bench 50%, 2× captain): {total_fp:.1f} pts")

# ── Save ──────────────────────────────────────────────────────
save_cols = [c for c in
             ['Player', 'Position', 'Team', 'price', 'fp_last_3',
              'predicted_fp', 'effective_fp', 'fantasy_points_mean',
              'predicted_value', 'is_starter', 'is_captain']
             if c in team_df.columns]
team_df[save_cols].to_csv('optimal_team_ml.csv', index=False)
print(f"\n✓ Saved to: optimal_team_ml.csv")

# ── High-upside alternatives ──────────────────────────────────
print("\n" + "=" * 60)
print("HIGH-UPSIDE ALTERNATIVES (not selected)")
print("=" * 60)

df['upside'] = df['predicted_fp'] - df['fantasy_points_mean']
alts = df[~df.index.isin(selected)].nlargest(5, 'upside')

print(f"\n{'Player':<25} {'Pos':<4} {'€':>5} {'Avg':>6} {'Pred':>6} {'Upside':>7}")
print("-" * 60)
for _, p in alts.iterrows():
    print(f"{p['Player']:<25} {p.get('Position','?'):<4} {p['price']:>5.1f} "
          f"{p['fantasy_points_mean']:>6.1f} {p['predicted_fp']:>6.1f} "
          f"{p['upside']:>7.1f}")

# ── Schedule: game times for selected round ───────────────────
if args.round is not None:
    try:
        from schedule_helper import load_el_schedule, detect_round, team_game_times
        sched = load_el_schedule(args.season)
        gameday = args.round
        tm = team_game_times(sched, gameday)

        first_game = min(tm.values()) if tm else None

        print("\n" + "=" * 60)
        print(f"GAME SCHEDULE — Round {gameday}")
        print("=" * 60)
        if first_game:
            print(f"\n  ⚠  Transfer deadline: {first_game.strftime('%a %b %d  %H:%M')}")
        print(f"\n  {'':2} {'':1} {'Player':<24}  {'Team':<5}  {'Game':<11}  {'Pred':>6}")
        print(f"  {'-'*54}")
        for _, p in team_df.sort_values(
                'effective_fp', ascending=False).iterrows():
            code = p['Team']
            game_str = tm[code].strftime('%a %H:%M') if code in tm else '?'
            cap  = '★' if p['is_captain'] else ' '
            role = 'S' if p['is_starter'] else 'B'
            print(f"  {cap} {role} {p['Player']:<24}  {code:<5}  {game_str:<11}  "
                  f"{p['predicted_fp']:>6.1f}")
    except Exception as e:
        print(f"\n(Schedule unavailable: {e})")

print("\n" + "=" * 60)
print("DONE! Ready to submit your team")
print("=" * 60)