"""
Optimize fantasy team using ML predictions.
Enforces 4G/4F/2C position constraints and picks an optimal captain (2× points).
Coach is co-optimised in the same ILP under the shared 100-credit budget.

Usage:
    python3 optimize_team_v2.py                 # no schedule info
    python3 optimize_team_v2.py --round 5       # D2-aware captain + game times
"""

import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, PULP_CBC_CMD

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--round',  type=int, default=None, help='Round number')
parser.add_argument('--season', type=int, default=2025)
args = parser.parse_args()

print("=" * 60)
print("EUROLEAGUE FANTASY OPTIMIZER V2 (ML-POWERED)")
print("=" * 60)

# ── Config ────────────────────────────────────────────────────
BUDGET       = 100
ROSTER       = 10
STARTERS     = 6
MAX_PER_TEAM = 6
REQ          = {'G': 4, 'F': 4, 'C': 2}
DNP_DISCOUNT = 0.5   # adj_fp = predicted_fp × (1 − DNP_DISCOUNT × dnp_rate_last5)

# ── Load predictions ──────────────────────────────────────────
df = pd.read_csv('data/processed/next_round_predictions.csv')
df = df[df['games_played_max'] >= 3].copy()
df = df.reset_index(drop=True)

# ── Attach position data ──────────────────────────────────────
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

# ── DNP risk discount ─────────────────────────────────────────
if 'dnp_rate_last5' in df.columns:
    df['adj_predicted_fp'] = df['predicted_fp'] * (
        1 - DNP_DISCOUNT * df['dnp_rate_last5']
    ).clip(lower=0)
    n_risky = (df['dnp_rate_last5'] > 0).sum()
    print(f"\nDNP discount ({DNP_DISCOUNT}×): {n_risky} players affected")
else:
    df['adj_predicted_fp'] = df['predicted_fp']
    print("\nWarning: dnp_rate_last5 not found — run build_features.py to enable DNP discount")

# ── Schedule fetch (before LP — needed for D2 captain constraint) ─────────────
tm          = {}       # {team_code: game_datetime}
d1_team_set = set()    # teams playing on the earliest game day
d2_team_set = set()

if args.round is not None:
    try:
        from schedule_helper import load_el_schedule, team_game_times
        sched    = load_el_schedule(args.season)
        tm       = team_game_times(sched, args.round)
        if tm:
            day1_time   = min(tm.values())
            d1_team_set = {t for t, dt in tm.items() if dt == day1_time}
            d2_team_set = {t for t, dt in tm.items() if dt > day1_time}
            print(f"\nSchedule: {len(d1_team_set)} D1 teams, {len(d2_team_set)} D2 teams")
    except Exception as e:
        print(f"\nSchedule unavailable ({e}) — captain constraint not applied")

# ── Coach data ────────────────────────────────────────────────
# Expected coach FP from win-rate + home-bonus model (same as coach_optimizer.py)
def _coach_fp_formula(margin, won):
    if won:
        return 10 if margin <= 10 else (20 if margin <= 20 else 25)
    return -5 if margin <= 10 else (-10 if margin <= 20 else -20)

coach_df = pd.DataFrame()   # empty = no coach in ILP

try:
    raw = pd.read_csv('data/raw/all_player_game_stats_2025.csv')
    el  = raw[(raw['Competition'] == 'Euroleague') & (raw['Player'] == 'Total')].copy()
    h   = el[el['Home'] == 1][['Round', 'Gamecode', 'Team', 'Points']].rename(
              columns={'Team': 'home_team', 'Points': 'home_pts'})
    a   = el[el['Home'] == 0][['Round', 'Gamecode', 'Team', 'Points']].rename(
              columns={'Team': 'away_team', 'Points': 'away_pts'})
    games = h.merge(a, on=['Round', 'Gamecode'])
    games['margin']   = (games['home_pts'] - games['away_pts']).abs()
    games['home_win'] = games['home_pts'] > games['away_pts']

    tg_rows = []
    for _, g in games.iterrows():
        tg_rows += [
            {'Round': g['Round'], 'team': g['home_team'],
             'won': g['home_win'],  'margin': g['margin']},
            {'Round': g['Round'], 'team': g['away_team'],
             'won': not g['home_win'], 'margin': g['margin']},
        ]
    tg = pd.DataFrame(tg_rows)
    tg['coach_fp'] = tg.apply(lambda r: _coach_fp_formula(int(r['margin']), bool(r['won'])), axis=1)

    target_round = args.round if args.round else tg['Round'].max() + 1
    prior = tg[tg['Round'] < target_round]
    rec   = prior.groupby('team').agg(
        wins=('won', 'sum'), games=('coach_fp', 'count'),
        fp_sum=('coach_fp', 'sum'), avg_margin=('margin', 'mean')
    ).copy()
    rec['win_rate'] = rec['wins'] / rec['games'].clip(lower=1)
    recent = prior.sort_values('Round').groupby('team').tail(5)
    rec_form = recent.groupby('team').agg(
        recent_wins=('won', 'sum'), recent_games=('coach_fp', 'count'))
    rec = rec.join(rec_form, how='left').fillna(0)
    rec['recent_win_rate'] = rec['recent_wins'] / rec['recent_games'].clip(lower=1)

    if tm:
        # Use teams actually playing this round
        playing_teams = list(tm.keys())
    else:
        playing_teams = list(rec.index)

    coach_rows = []
    for team in playing_teams:
        if team in rec.index:
            r = rec.loc[team]
        else:
            r = pd.Series({'win_rate': 0.5, 'recent_win_rate': 0.5, 'fp_sum': 0})
        is_home   = team in d1_team_set or (not d1_team_set and True)
        if tm:
            opp_list  = [t for t in tm.keys() if t != team]
            is_home   = team in d1_team_set
        else:
            opp_list  = []
        opp_wr = 0.5
        if opp_list and opp_list[0] in rec.index:
            opp_wr = rec.loc[opp_list[0], 'win_rate']

        strength  = r['win_rate'] * 0.5 + r['recent_win_rate'] * 0.5
        net       = strength - opp_wr
        win_prob  = float(np.clip(0.5 + net + (0.08 if is_home else 0.0), 0.05, 0.95))
        exp_fp    = (win_prob * (10*0.45 + 20*0.38 + 25*0.17) +
                     (1-win_prob) * (-5*0.57 - 10*0.29 - 20*0.14))
        coach_rows.append({'team': team, 'exp_coach_fp': round(exp_fp, 2),
                           'win_prob': round(win_prob*100, 1),
                           'win_rate': round(r['win_rate']*100, 1)})

    coach_df = pd.DataFrame(coach_rows).set_index('team')

    # Attach prices
    if Path('coach_prices.csv').exists():
        cp = pd.read_csv('coach_prices.csv')
        cp = cp.set_index('team')[['price']].rename(columns={'price': 'coach_price'})
        coach_df = coach_df.join(cp, how='inner')
        print(f"Coach prices loaded: {len(coach_df)} coaches eligible")
    else:
        print("\nWarning: coach_prices.csv not found — run scrape_prices.py to enable coach co-optimisation")
        print("         Coach excluded from budget for this run.")
        coach_df = pd.DataFrame()   # no coach in LP

except Exception as e:
    print(f"\nCoach data unavailable ({e}) — skipping coach selection")
    coach_df = pd.DataFrame()

# ── Position feasibility ──────────────────────────────────────
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
if not coach_df.empty:
    print(f"  Coach: co-optimised ({len(coach_df)} candidates)")
if d2_team_set:
    d2_indices = df[df['Team'].isin(d2_team_set)].index.tolist()
    print(f"  Captain: restricted to {len(d2_team_set)} Day-2 teams ({len(d2_indices)} eligible players)")

# ── Build LP problem ──────────────────────────────────────────
def build_prob(name, use_positions):
    prob = LpProblem(name, LpMaximize)
    pv = {i: LpVariable(f"p_{i}", cat='Binary') for i in df.index}
    sv = {i: LpVariable(f"s_{i}", cat='Binary') for i in df.index}
    cv = {i: LpVariable(f"c_{i}", cat='Binary') for i in df.index}

    # Coach binary vars (only if prices available)
    kv = {}
    if not coach_df.empty:
        kv = {t: LpVariable(f"k_{t}", cat='Binary') for t in coach_df.index}

    fp = df['adj_predicted_fp']
    prob += lpSum(
        fp[i] * 0.5 * pv[i] +
        fp[i] * 0.5 * sv[i] +
        fp[i] * cv[i]
        for i in df.index
    ) + lpSum(
        coach_df.loc[t, 'exp_coach_fp'] * kv[t] for t in kv
    )

    prob += lpSum(pv[i] for i in df.index) == ROSTER
    prob += lpSum(sv[i] for i in df.index) == STARTERS

    # Budget: players + coach
    player_cost = lpSum(pv[i] * df.loc[i, 'price'] for i in df.index)
    coach_cost  = lpSum(coach_df.loc[t, 'coach_price'] * kv[t] for t in kv)
    prob += player_cost + coach_cost <= BUDGET

    for team, grp in df.groupby('Team'):
        prob += lpSum(pv[i] for i in grp.index) <= MAX_PER_TEAM
    if use_positions:
        for pos, req in REQ.items():
            pos_idx = df[df['Position'] == pos].index
            prob += lpSum(pv[i] for i in pos_idx) == req

    prob += lpSum(cv[i] for i in df.index) == 1
    for i in df.index:
        prob += sv[i] <= pv[i]
        prob += cv[i] <= sv[i]

    if kv:
        prob += lpSum(kv[t] for t in kv) == 1

    # D2 captain constraint: captain must be a Day-2 player
    if d2_team_set:
        d2_idx = set(df[df['Team'].isin(d2_team_set)].index)
        if d2_idx:
            for i in df.index:
                if i not in d2_idx:
                    prob += cv[i] == 0

    return prob, pv, sv, cv, kv

print("\nOptimizing...")
prob, player_vars, starter_vars, captain_vars, coach_vars = build_prob("Fantasy_Team_ML", has_positions)
prob.solve(PULP_CBC_CMD(msg=0))

if prob.status != 1:
    print(f"\nWARNING: Infeasible (status {prob.status})")
    if has_positions:
        print("Retrying without position constraints...")
        prob, player_vars, starter_vars, captain_vars, coach_vars = build_prob("Fantasy_Team_ML_NoPos", False)
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
sel_coach = next((t for t, v in coach_vars.items() if v.varValue == 1), None) if coach_vars else None

team_df = df.loc[selected].copy()
team_df['is_captain'] = team_df.index == captain
team_df['is_starter'] = [starter_vars[i].varValue == 1 for i in team_df.index]
team_df['effective_fp'] = np.where(
    team_df['is_captain'],
    team_df['predicted_fp'] * 2,
    np.where(team_df['is_starter'], team_df['predicted_fp'], team_df['predicted_fp'] * 0.5)
)
team_df = team_df.sort_values('effective_fp', ascending=False)

# ── Print team ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("OPTIMAL TEAM")
print("=" * 60)

has_dnp = 'dnp_rate_last5' in team_df.columns
hdr = f"\n{'★':<2} {'S':<2} {'Player':<25} {'Pos':<4} {'Team':<5} {'€':>5} {'L3':>6} {'Eff':>6} {'Avg':>6}"
if has_dnp:
    hdr += f" {'DNP%':>6}"
print(hdr)
print("-" * (76 + (7 if has_dnp else 0)))

total_cost = total_fp = total_avg = 0
for _, p in team_df.iterrows():
    cap   = "★" if p['is_captain'] else " "
    start = "S" if p['is_starter'] else "B"
    pos   = p.get('Position', '?')
    line  = (f"{cap:<2} {start:<2} {p['Player']:<25} {pos:<4} {p['Team']:<5} "
             f"{p['price']:>5.1f} {p['fp_last_3']:>6.1f} "
             f"{p['effective_fp']:>6.1f} {p['fantasy_points_mean']:>6.1f}")
    if has_dnp:
        line += f" {p['dnp_rate_last5']*100:>5.0f}%"
    print(line)
    total_cost += p['price']
    total_fp   += p['effective_fp']
    total_avg  += p['fantasy_points_mean']

coach_price_used = coach_df.loc[sel_coach, 'coach_price'] if sel_coach else 0
print("-" * (76 + (7 if has_dnp else 0)))
print(f"{'TOTAL (players)':<35} {total_cost:>5.1f} {'':>6} {total_fp:>6.1f} {total_avg:>6.1f}")

if captain is not None:
    cap_row = df.loc[captain]
    role    = "(Day-2 ✓)" if cap_row['Team'] in d2_team_set else ""
    print(f"\n★ Captain: {cap_row['Player']}  "
          f"({cap_row['predicted_fp']:.1f} → {cap_row['predicted_fp']*2:.1f} pts) {role}")

budget_used = total_cost + coach_price_used
print(f"  Players: {total_cost:.1f}  Coach: {coach_price_used:.1f}  Total: {budget_used:.1f} / {BUDGET}  "
      f"(Remaining: {BUDGET - budget_used:.1f})")
print(f"  Expected total: {total_fp:.1f} pts")

# ── Coach section ─────────────────────────────────────────────
if sel_coach:
    cr = coach_df.loc[sel_coach]
    print("\n" + "=" * 60)
    print("COACH (co-optimised)")
    print("=" * 60)
    print(f"\n  ★ {sel_coach}  —  win prob {cr['win_prob']:.1f}%  "
          f"exp FP {cr['exp_coach_fp']:.1f}  price €{cr['coach_price']:.1f}")
    print(f"\n  {'Team':<6} {'Win%':>5} {'Exp FP':>7} {'€':>5}")
    print(f"  {'-'*26}")
    for t, row in coach_df.sort_values('exp_coach_fp', ascending=False).iterrows():
        marker = "★" if t == sel_coach else " "
        print(f"  {marker} {t:<5} {row['win_prob']:>5.1f} {row['exp_coach_fp']:>7.1f} "
              f"{row['coach_price']:>5.1f}")
elif not coach_df.empty:
    print("\n(No coach selected — budget may be too tight)")
else:
    print("\nTip: Run scrape_prices.py to populate coach_prices.csv and enable coach co-optimisation.")

# ── DNP risk table ────────────────────────────────────────────
if has_dnp:
    print("\n" + "=" * 60)
    print("DNP RISK — SELECTED TEAM")
    print("=" * 60)
    print(f"\n  {'Player':<25} {'DNP-L5':>6} {'DNP-L10':>7} {'Pred FP':>8} {'Adj FP':>8}")
    print(f"  {'-'*57}")
    for _, p in team_df.sort_values('dnp_rate_last5', ascending=False).iterrows():
        adj = p['predicted_fp'] * max(0, 1 - DNP_DISCOUNT * p['dnp_rate_last5'])
        print(f"  {p['Player']:<25} {p['dnp_rate_last5']*100:>5.0f}%  "
              f"{p['dnp_rate_last10']*100:>6.0f}%  "
              f"{p['predicted_fp']:>8.1f}  {adj:>8.1f}")

# ── Save ──────────────────────────────────────────────────────
save_cols = [c for c in
             ['Player', 'Position', 'Team', 'price', 'fp_last_3',
              'predicted_fp', 'adj_predicted_fp', 'effective_fp',
              'fantasy_points_mean', 'predicted_value',
              'dnp_rate_last5', 'dnp_rate_last10', 'is_starter', 'is_captain']
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

# ── Schedule: game times ──────────────────────────────────────
if args.round is not None and tm:
    first_game = min(tm.values())
    print("\n" + "=" * 60)
    print(f"GAME SCHEDULE — Round {args.round}")
    print("=" * 60)
    print(f"\n  ⚠  Transfer deadline: {first_game.strftime('%a %b %d  %H:%M')}")
    if d2_team_set:
        print(f"  Captain locked to Day-2 players (decide after Day-1 results)")
    print(f"\n  {'':2} {'':1} {'Player':<24}  {'Team':<5}  {'Game':<11}  {'Pred':>6}  {'Day'}")
    print(f"  {'-'*62}")
    for _, p in team_df.sort_values('effective_fp', ascending=False).iterrows():
        code     = p['Team']
        game_str = tm[code].strftime('%a %H:%M') if code in tm else '?'
        day_tag  = 'D1' if code in d1_team_set else ('D2' if code in d2_team_set else '?')
        cap      = '★' if p['is_captain'] else ' '
        role     = 'S' if p['is_starter'] else 'B'
        print(f"  {cap} {role} {p['Player']:<24}  {code:<5}  {game_str:<11}  "
              f"{p['predicted_fp']:>6.1f}  {day_tag}")

print("\n" + "=" * 60)
print("DONE! Ready to submit your team")
print("=" * 60)