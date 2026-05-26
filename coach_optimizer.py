"""
coach_optimizer.py — EuroLeague Fantasy coach picker.

Scores each team's coach FP from historical margins and predicts the best
coach to pick for the upcoming round using team standing + recent form.

Scoring table (official):
  Win by 1-10 or OT  → +10
  Win by 11-20       → +20
  Win by 20+         → +25
  Loss by 1-10 or OT → -5
  Loss by 11-20      → -10
  Loss by 20+        → -20

Usage:
    python3 coach_optimizer.py                   # next round prediction
    python3 coach_optimizer.py --round 15        # specific round
    python3 coach_optimizer.py --history         # full season oracle breakdown
"""

import argparse
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

SEASON = 2025

parser = argparse.ArgumentParser()
parser.add_argument('--round',   type=int, default=None)
parser.add_argument('--history', action='store_true', help='Show full season breakdown')
args = parser.parse_args()


# ── Coach FP formula ──────────────────────────────────────────────────────────
def coach_fp(margin: int, won: bool) -> int:
    if won:
        return 10 if margin <= 10 else (20 if margin <= 20 else 25)
    else:
        return -5 if margin <= 10 else (-10 if margin <= 20 else -20)


# ── Load game margins from cached boxscore totals ─────────────────────────────
print("Loading game data…")
raw = pd.read_csv('data/raw/all_player_game_stats_2025.csv')
el  = raw[(raw['Competition'] == 'Euroleague') & (raw['Player'] == 'Total')].copy()

home = el[el['Home'] == 1][['Round', 'Gamecode', 'Team', 'Points']].rename(
    columns={'Team': 'home_team', 'Points': 'home_pts'})
away = el[el['Home'] == 0][['Round', 'Gamecode', 'Team', 'Points']].rename(
    columns={'Team': 'away_team', 'Points': 'away_pts'})
games = home.merge(away, on=['Round', 'Gamecode'])
games['margin']   = (games['home_pts'] - games['away_pts']).abs()
games['home_win'] = games['home_pts'] > games['away_pts']
games['winner']   = games.apply(lambda r: r['home_team'] if r['home_win'] else r['away_team'], axis=1)
games['loser']    = games.apply(lambda r: r['away_team'] if r['home_win'] else r['home_team'], axis=1)

# Flat table: one row per team per game
home_fp = games[['Round', 'Gamecode', 'home_team', 'home_win', 'margin']].copy()
home_fp['team']     = home_fp['home_team']
home_fp['is_home']  = True
home_fp['won']      = home_fp['home_win']
away_fp = games[['Round', 'Gamecode', 'away_team', 'home_win', 'margin']].copy()
away_fp['team']    = away_fp['away_team']
away_fp['is_home'] = False
away_fp['won']     = ~away_fp['home_win']

team_games = pd.concat([
    home_fp[['Round', 'Gamecode', 'team', 'is_home', 'won', 'margin']],
    away_fp[['Round', 'Gamecode', 'team', 'is_home', 'won', 'margin']],
], ignore_index=True)
team_games['coach_fp'] = team_games.apply(
    lambda r: coach_fp(int(r['margin']), bool(r['won'])), axis=1)


# ── Season stats per team ─────────────────────────────────────────────────────
season_stats = (team_games.groupby('team')
                .agg(games=('coach_fp', 'count'),
                     wins=('won', 'sum'),
                     total_fp=('coach_fp', 'sum'),
                     avg_fp=('coach_fp', 'mean'),
                     avg_margin=('margin', 'mean'))
                .sort_values('total_fp', ascending=False))


# ── Round-by-round coach prediction ───────────────────────────────────────────
def predict_round(target_round: int) -> pd.DataFrame:
    """
    Score each team playing in target_round using cumulative record up to
    round target_round-1 + home-court advantage.
    Returns ranked DataFrame.
    """
    from euroleague_api.schedule import Schedule

    prior = team_games[team_games['Round'] < target_round]
    record = prior.groupby('team').agg(
        wins=('won', 'sum'),
        games=('coach_fp', 'count'),
        fp_sum=('coach_fp', 'sum'),
        avg_margin=('margin', 'mean'),
    ).copy()
    record['win_rate'] = record['wins'] / record['games'].clip(lower=1)
    # Recent form: last 5 games
    recent = prior.sort_values('Round').groupby('team').tail(5)
    recent_form = recent.groupby('team').agg(
        recent_wins=('won', 'sum'),
        recent_games=('coach_fp', 'count'),
    )
    record = record.join(recent_form, how='left').fillna(0)
    record['recent_win_rate'] = record['recent_wins'] / record['recent_games'].clip(lower=1)

    # Get schedule for the round
    sched = Schedule('E').get_schedule(SEASON)
    r = sched[sched['gameday'] == target_round]

    rows = []
    for _, game in r.iterrows():
        for code, is_home in [(game['homecode'], True), (game['awaycode'], False)]:
            if code in record.index:
                rec = record.loc[code]
            else:
                rec = pd.Series({'win_rate': 0.5, 'recent_win_rate': 0.5,
                                  'avg_margin': 5.0, 'fp_sum': 0})
            opponent = game['awaycode'] if is_home else game['homecode']
            opp_win_rate = record.loc[opponent, 'win_rate'] if opponent in record.index else 0.5
            # Expected coach FP: blend win rate × expected margin × home bonus
            strength = rec['win_rate'] * 0.5 + rec['recent_win_rate'] * 0.5
            opp_strength = opp_win_rate
            net = strength - opp_strength   # > 0 = favoured
            home_bonus = 0.08 if is_home else 0.0
            win_prob = min(0.95, max(0.05, 0.5 + net + home_bonus))
            # Expected FP under win probability and typical margins
            exp_fp = (win_prob * (10 * 0.45 + 20 * 0.38 + 25 * 0.17) +
                      (1 - win_prob) * (-5 * 0.57 - 10 * 0.29 - 20 * 0.14))
            rows.append({
                'team': code,
                'opponent': opponent,
                'is_home': is_home,
                'win_rate': round(rec['win_rate'] * 100, 1),
                'recent_win_rate': round(rec['recent_win_rate'] * 100, 1),
                'win_prob': round(win_prob * 100, 1),
                'exp_coach_fp': round(exp_fp, 1),
                'season_coach_fp': int(rec.get('fp_sum', 0)),
            })

    return pd.DataFrame(rows).sort_values('exp_coach_fp', ascending=False)


# ── Determine target round ────────────────────────────────────────────────────
all_rounds = sorted(team_games['Round'].unique())

if args.round is not None:
    target_round = args.round
else:
    # Next unplayed round = last round + 1 (season over → last round)
    target_round = all_rounds[-1]

print(f"\n{'='*60}")
print(f"  COACH RECOMMENDATION — Round {target_round}")
print(f"{'='*60}")
print(f"\n  {'Team':<6} {'Opp':<6} {'H/A':<4} {'WR%':>5} {'L5 WR%':>7} {'Win%':>6} {'Exp FP':>7}")
print(f"  {'-'*52}")

try:
    picks = predict_round(target_round)
    for _, p in picks.iterrows():
        ha = 'H' if p['is_home'] else 'A'
        print(f"  {p['team']:<6} vs {p['opponent']:<4} {ha:<4} "
              f"{p['win_rate']:>5.1f} {p['recent_win_rate']:>7.1f} "
              f"{p['win_prob']:>6.1f} {p['exp_coach_fp']:>7.1f}")
    best = picks.iloc[0]
    print(f"\n  ★ Recommended: {best['team']}  "
          f"(vs {best['opponent']}, {'home' if best['is_home'] else 'away'})  "
          f"exp FP = {best['exp_coach_fp']:.1f}")
except Exception as e:
    print(f"  (Schedule lookup failed: {e})")


# ── Historical breakdown ───────────────────────────────────────────────────────
if args.history or True:   # always show season summary
    print(f"\n{'='*60}")
    print("  SEASON COACH FP SUMMARY (oracle = best pick each round)")
    print(f"{'='*60}")

    oracle_by_round = team_games.groupby('Round').apply(
        lambda g: g.loc[g['coach_fp'].idxmax()], include_groups=False)
    oracle_fp = team_games.groupby('Round')['coach_fp'].max()

    print(f"\n  Oracle coach avg:   {oracle_fp.mean():.1f} pts/round")
    print(f"  Oracle coach total: {oracle_fp.sum():.0f} pts over {len(oracle_fp)} rounds")
    print(f"\n  Coach FP distribution (all team-games):")
    vc = team_games['coach_fp'].value_counts().sort_index()
    total = len(team_games)
    for v, cnt in vc.items():
        print(f"    {v:>4} pts: {cnt:>3}×  ({cnt/total*100:.0f}%)")

    if args.history:
        print(f"\n  {'Round':>6}  {'Best team':>6}  {'Coach FP':>9}")
        print(f"  {'-'*28}")
        for rnd in sorted(oracle_fp.index):
            best_row = team_games[(team_games['Round'] == rnd)].nlargest(1, 'coach_fp').iloc[0]
            print(f"  {rnd:>6}  {best_row['team']:>6}  {int(best_row['coach_fp']):>9}")

    print(f"\n  Top coaches by season total FP (if you'd picked them all season):")
    print(f"  {'Team':<6} {'Games':>6} {'Wins':>5} {'Total FP':>9} {'Avg FP':>7}")
    print(f"  {'-'*38}")
    for team, row in season_stats.head(10).iterrows():
        print(f"  {team:<6} {int(row['games']):>6} {int(row['wins']):>5} "
              f"{int(row['total_fp']):>9} {row['avg_fp']:>7.1f}")