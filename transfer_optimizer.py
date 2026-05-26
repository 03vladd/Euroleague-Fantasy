"""
Transfer optimizer — suggests the best 1-3 swaps from your current squad.

Reads my_team.json (your actual squad) and next_round_predictions.csv,
then finds which player(s) to sell and who to buy to maximise expected FP.

Usage:
    python3 transfer_optimizer.py                  # suggest best 1 transfer
    python3 transfer_optimizer.py --transfers 2    # suggest best 2-transfer combo
    python3 transfer_optimizer.py --round 5        # also show schedule context
"""

import argparse
import json
import warnings
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--transfers', type=int, default=1, help='Number of transfers to make (1-3)')
parser.add_argument('--round',     type=int, default=None)
parser.add_argument('--season',    type=int, default=2025)
args = parser.parse_args()

BUDGET       = 100
ROSTER       = 10
STARTERS     = 6
MAX_PER_TEAM = 6
REQ          = {'G': 4, 'F': 4, 'C': 2}
DNP_DISCOUNT        = 0.5
CAPTAIN_STD_PENALTY = 0.3

print("=" * 60)
print("EUROLEAGUE TRANSFER OPTIMIZER")
print("=" * 60)

# ── Load current squad ────────────────────────────────────────
team_file = Path('my_team.json')
if not team_file.exists():
    print("\nERROR: my_team.json not found.")
    print("Run the main optimizer first or create my_team.json manually.")
    raise SystemExit(1)

with open(team_file) as f:
    my_team = json.load(f)

current_players = [p for p in my_team.get('players', []) if p]
current_coach   = my_team.get('coach')
current_round   = my_team.get('round')

if len(current_players) != ROSTER:
    print(f"\nERROR: my_team.json has {len(current_players)} players, expected {ROSTER}.")
    print("Edit my_team.json with your actual 10-player squad.")
    raise SystemExit(1)

print(f"\nCurrent squad ({len(current_players)} players, round {current_round}):")
for p in current_players:
    print(f"  {p}")
print(f"Coach: {current_coach or '(none)'}")

# ── Load predictions ──────────────────────────────────────────
pred_file = Path('data/processed/next_round_predictions.csv')
if not pred_file.exists():
    print("\nERROR: next_round_predictions.csv not found. Run train_model.py first.")
    raise SystemExit(1)

df = pd.read_csv(pred_file)
df = df[df['games_played_max'] >= 3].copy().reset_index(drop=True)

# Attach positions
POSITION_MAP = {'Guard': 'G', 'Forward': 'F', 'Center': 'C'}
try:
    pos_df = (pd.read_csv('euroleague_fantasy_analysis_complete.csv')
              [['player_name', 'estimated_position']]
              .drop_duplicates('player_name')
              .rename(columns={'player_name': 'Player', 'estimated_position': 'Position'}))
    pos_df['Position'] = pos_df['Position'].map(POSITION_MAP).fillna('?')
    df = df.merge(pos_df, on='Player', how='left')
    df['Position'] = df['Position'].fillna('?')
except FileNotFoundError:
    df['Position'] = '?'

# DNP discount + captain reliability
if 'dnp_rate_last5' in df.columns:
    df['adj_predicted_fp'] = (df['predicted_fp'] * (1 - DNP_DISCOUNT * df['dnp_rate_last5'])
                              .clip(lower=0))
else:
    df['adj_predicted_fp'] = df['predicted_fp']

if 'fp_std_5' in df.columns:
    df['captain_score'] = (df['adj_predicted_fp']
                           - CAPTAIN_STD_PENALTY * df['fp_std_5']).clip(lower=0)
else:
    df['captain_score'] = df['adj_predicted_fp']

# ── Score a squad ─────────────────────────────────────────────
def score_squad(players: list[str]) -> float:
    """Expected FP for a squad using LP-optimal starter/captain assignment."""
    pool = df[df['Player'].isin(players)].copy()
    if len(pool) < ROSTER:
        return -999.0
    top6 = pool.nlargest(STARTERS, 'adj_predicted_fp')
    cap  = pool.loc[top6['captain_score'].idxmax()]
    total = 0.0
    for _, row in pool.iterrows():
        fp = row['adj_predicted_fp']
        if row['Player'] == cap['Player']:
            total += fp * 2
        elif row['Player'] in top6['Player'].values:
            total += fp
        else:
            total += fp * 0.5
    return total

# ── Current squad score ───────────────────────────────────────
current_in_pool = [p for p in current_players if p in df['Player'].values]
missing         = [p for p in current_players if p not in df['Player'].values]
if missing:
    print(f"\nWarning: {len(missing)} player(s) not in predictions (no price or <3 games):")
    for p in missing:
        print(f"  {p}")

current_fp = score_squad(current_players)
current_squad = df[df['Player'].isin(current_players)].copy()
current_price = current_squad['price'].sum()

print(f"\nCurrent squad expected FP: {current_fp:.1f}")
print(f"Current squad price:        {current_price:.1f}")

# Available budget for transfers = BUDGET - current price
# (selling a player frees their price, buying costs the new player's price)

# ── Find best N transfers ─────────────────────────────────────
n = min(args.transfers, 3)
print(f"\nSearching best {n}-transfer combo...")

candidates = df[~df['Player'].isin(current_players)].copy()
# Pre-filter candidates to only consider players worth picking up
candidates = candidates.nlargest(60, 'adj_predicted_fp')

best_gain    = 0.0
best_out     = []
best_in      = []
best_new_squad = current_players[:]

sell_combos = list(itertools.combinations(current_players, n))
buy_combos  = list(itertools.combinations(candidates['Player'].tolist(), n))

checked = 0
for sell in sell_combos:
    sell_value = df[df['Player'].isin(sell)]['price'].sum() if all(
        p in df['Player'].values for p in sell) else 0

    temp_squad = [p for p in current_players if p not in sell]
    temp_price = df[df['Player'].isin(temp_squad)]['price'].sum()
    budget_for_buys = BUDGET - temp_price

    for buy in buy_combos:
        buy_cost = candidates[candidates['Player'].isin(buy)]['price'].sum()
        if buy_cost > budget_for_buys:
            continue

        new_squad = temp_squad + list(buy)

        # Quick team cap check
        team_counts = df[df['Player'].isin(new_squad)]['Team'].value_counts()
        if team_counts.max() > MAX_PER_TEAM:
            continue

        gain = score_squad(new_squad) - current_fp
        checked += 1
        if gain > best_gain:
            best_gain      = gain
            best_out       = list(sell)
            best_in        = list(buy)
            best_new_squad = new_squad

print(f"Evaluated {checked:,} combos")

# ── Print results ─────────────────────────────────────────────
print("\n" + "=" * 60)
if best_gain <= 0:
    print("No improvement found — your squad is already optimal!")
    print(f"Current squad expected FP: {current_fp:.1f}")
else:
    new_fp    = current_fp + best_gain
    new_price = df[df['Player'].isin(best_new_squad)]['price'].sum()
    print(f"BEST {n}-TRANSFER PLAN  (+{best_gain:.1f} expected FP)")
    print("=" * 60)
    for out, ins in zip(best_out, best_in):
        out_fp  = df[df['Player'] == out]['adj_predicted_fp'].values
        in_fp   = df[df['Player'] == ins]['adj_predicted_fp'].values
        out_pr  = df[df['Player'] == out]['price'].values
        in_pr   = df[df['Player'] == ins]['price'].values
        out_fp  = out_fp[0] if len(out_fp) else 0
        in_fp   = in_fp[0]  if len(in_fp)  else 0
        out_pr  = out_pr[0] if len(out_pr) else 0
        in_pr   = in_pr[0]  if len(in_pr)  else 0
        delta_p = in_pr - out_pr
        print(f"\n  OUT: {out:<28}  FP {out_fp:>5.1f}  €{out_pr:.1f}")
        print(f"  IN:  {ins:<28}  FP {in_fp:>5.1f}  €{in_pr:.1f}  (Δ€{delta_p:+.1f})")

    print(f"\n  Before: {current_fp:.1f} FP  |  After: {new_fp:.1f} FP  (+{best_gain:.1f})")
    print(f"  Budget: €{new_price:.1f} / {BUDGET}  (remaining: €{BUDGET - new_price:.1f})")

    print(f"\n{'─'*60}")
    print("NEW SQUAD AFTER TRANSFER:")
    print(f"{'─'*60}")
    new_squad_df = df[df['Player'].isin(best_new_squad)].sort_values(
        'adj_predicted_fp', ascending=False)
    new_squad_df['role'] = ''
    top6_idx = new_squad_df.nlargest(STARTERS, 'adj_predicted_fp').index
    cap_idx  = new_squad_df.loc[top6_idx, 'captain_score'].idxmax()
    new_squad_df.loc[top6_idx, 'role'] = 'S'
    new_squad_df.loc[cap_idx,  'role'] = '★S'
    new_squad_df['role'] = new_squad_df['role'].replace('', 'B')
    new_squad_df['tag'] = new_squad_df['Player'].apply(
        lambda p: ' ← NEW' if p in best_in else ('  ← STAY' if p not in best_out else ''))
    print(f"\n  {'':3} {'Player':<28} {'Pos':<4} {'Team':<5} {'€':>5} {'AdjFP':>6}")
    print(f"  {'-'*55}")
    for _, row in new_squad_df.iterrows():
        print(f"  {row['role']:<3} {row['Player']:<28} {row['Position']:<4} "
              f"{row['Team']:<5} {row['price']:>5.1f} {row['adj_predicted_fp']:>6.1f}"
              f"{row['tag']}")

print(f"\n{'=' * 60}")
print("To lock in this squad, update my_team.json with the new players.")
print("=" * 60)