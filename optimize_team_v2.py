"""
Optimize fantasy team using ML predictions.
Enforces 4G/4F/2C position constraints and picks an optimal captain (2× points).
"""

import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD

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
MAX_PER_TEAM   = 3
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
prob = LpProblem("Fantasy_Team_ML", LpMaximize)

player_vars  = {i: LpVariable(f"p_{i}", cat='Binary') for i in df.index}
captain_vars = {i: LpVariable(f"c_{i}", cat='Binary') for i in df.index}

# Objective: each player's FP + captain doubles their score (so add their FP once more)
prob += lpSum(
    player_vars[i] * df.loc[i, 'predicted_fp'] +
    captain_vars[i] * df.loc[i, 'predicted_fp']
    for i in df.index
)

# Roster size
prob += lpSum(player_vars[i] for i in df.index) == ROSTER

# Budget
prob += lpSum(player_vars[i] * df.loc[i, 'price'] for i in df.index) <= BUDGET

# Max 3 per team
for team, grp in df.groupby('Team'):
    prob += lpSum(player_vars[i] for i in grp.index) <= MAX_PER_TEAM

# Position requirements
if has_positions:
    for pos, req in REQ.items():
        pos_idx = df[df['Position'] == pos].index
        prob += lpSum(player_vars[i] for i in pos_idx) == req

# Captain: exactly 1, must be on the team
prob += lpSum(captain_vars[i] for i in df.index) == 1
for i in df.index:
    prob += captain_vars[i] <= player_vars[i]

# ── Solve ─────────────────────────────────────────────────────
print("\nOptimizing...")
prob.solve(PULP_CBC_CMD(msg=0))

if prob.status != 1:
    print(f"\nWARNING: Infeasible (status {prob.status})")
    if has_positions:
        print("Retrying without position constraints...")
        # Remove position constraints and retry
        prob2 = LpProblem("Fantasy_Team_ML_NoPos", LpMaximize)
        prob2 += lpSum(
            player_vars[i] * df.loc[i, 'predicted_fp'] +
            captain_vars[i] * df.loc[i, 'predicted_fp']
            for i in df.index
        )
        prob2 += lpSum(player_vars[i] for i in df.index) == ROSTER
        prob2 += lpSum(player_vars[i] * df.loc[i, 'price'] for i in df.index) <= BUDGET
        for team, grp in df.groupby('Team'):
            prob2 += lpSum(player_vars[i] for i in grp.index) <= MAX_PER_TEAM
        prob2 += lpSum(captain_vars[i] for i in df.index) == 1
        for i in df.index:
            prob2 += captain_vars[i] <= player_vars[i]
        prob2.solve(PULP_CBC_CMD(msg=0))
        if prob2.status != 1:
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
team_df['effective_fp'] = np.where(team_df['is_captain'],
                                   team_df['predicted_fp'] * 2,
                                   team_df['predicted_fp'])
team_df = team_df.sort_values('effective_fp', ascending=False)

print("\n" + "=" * 60)
print("OPTIMAL TEAM")
print("=" * 60)
print(f"\n{'★':<2} {'Player':<25} {'Pos':<4} {'Team':<5} {'€':>5} {'L3':>6} {'Pred':>6} {'Avg':>6}")
print("-" * 72)

total_cost = total_fp = total_avg = 0
for _, p in team_df.iterrows():
    cap  = "★" if p['is_captain'] else " "
    pos  = p.get('Position', '?')
    print(f"{cap:<2} {p['Player']:<25} {pos:<4} {p['Team']:<5} "
          f"{p['price']:>5.1f} {p['fp_last_3']:>6.1f} "
          f"{p['effective_fp']:>6.1f} {p['fantasy_points_mean']:>6.1f}")
    total_cost += p['price']
    total_fp   += p['effective_fp']
    total_avg  += p['fantasy_points_mean']

print("-" * 72)
print(f"{'TOTAL':<33} {total_cost:>5.1f} {'':>6} {total_fp:>6.1f} {total_avg:>6.1f}")

if captain is not None:
    cap_row = df.loc[captain]
    print(f"\n★ Captain: {cap_row['Player']}  "
          f"({cap_row['predicted_fp']:.1f} → {cap_row['predicted_fp']*2:.1f} pts)")

print(f"  Budget remaining: {BUDGET - total_cost:.1f} credits")
print(f"  Expected total (incl. 2× captain): {total_fp:.1f} pts")

# ── Save ──────────────────────────────────────────────────────
save_cols = [c for c in
             ['Player', 'Position', 'Team', 'price', 'fp_last_3',
              'predicted_fp', 'effective_fp', 'fantasy_points_mean',
              'predicted_value', 'is_captain']
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

print("\n" + "=" * 60)
print("DONE! Ready to submit your team")
print("=" * 60)