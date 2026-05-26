"""
schedule_helper.py — EuroLeague game schedule + transfer window info.

Shows the game dates for a given round and maps each team to its game time
so the optimizer knows when each player plays. EuroLeague-only — Eurocup
schedule is not relevant for team selection.

Usage:
    python3 schedule_helper.py                  # auto-detect next/current round
    python3 schedule_helper.py --round 5        # specific round
    python3 schedule_helper.py --season 2025    # default season (2025-26)
"""

import argparse
from datetime import datetime, timedelta

import pandas as pd
from euroleague_api.schedule import Schedule

SEASON = 2025

parser = argparse.ArgumentParser()
parser.add_argument('--round',  type=int, default=None)
parser.add_argument('--season', type=int, default=SEASON)
args = parser.parse_args()


def parse_gametime(date_str: str, time_str: str) -> datetime | None:
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%b %d, %Y %H:%M")
    except Exception:
        return None


def load_el_schedule(season: int) -> pd.DataFrame:
    sched = Schedule('E').get_schedule(season)
    sched['gametime'] = sched.apply(
        lambda r: parse_gametime(r['date'], r['startime']), axis=1
    )
    return sched.dropna(subset=['gametime'])


def detect_round(sched: pd.DataFrame) -> int:
    """Return the round that contains today, or the next upcoming round."""
    now = datetime.now()
    future = sched[sched['gametime'] >= now]
    if future.empty:
        return int(sched['gameday'].max())
    return int(future.nsmallest(1, 'gametime')['gameday'].iloc[0])


def team_game_times(sched: pd.DataFrame, gameday: int) -> dict[str, datetime]:
    """Return {team_code: game_datetime} for every team in the round."""
    r = sched[sched['gameday'] == gameday]
    mapping: dict[str, datetime] = {}
    for _, row in r.iterrows():
        for code in (row['homecode'], row['awaycode']):
            if code not in mapping or row['gametime'] < mapping[code]:
                mapping[code] = row['gametime']
    return mapping


def print_round(sched: pd.DataFrame, gameday: int):
    r = sched[sched['gameday'] == gameday].sort_values('gametime').reset_index(drop=True)
    if r.empty:
        print(f"No games found for round {gameday}.")
        return

    first_game = r['gametime'].min()

    print(f"\n{'='*60}")
    print(f"  EuroLeague — Round {gameday}")
    print(f"{'='*60}")

    r['day'] = r['gametime'].dt.strftime('%a %b %d')
    for day_label, grp in r.groupby('day', sort=False):
        print(f"\n  {day_label}")
        print(f"  {'Time':<7}  {'Home':<28}  {'Away'}")
        print(f"  {'-'*52}")
        for _, g in grp.iterrows():
            print(f"  {g['startime']:<7}  {g['hometeam']:<28}  {g['awayteam']}")

    print(f"\n  ⚠  Transfer deadline: {first_game.strftime('%a %b %d  %H:%M')}  (first tipoff of round)")


def print_team_with_schedule(sched: pd.DataFrame, gameday: int):
    """Annotate optimal_team_ml.csv with game times for round `gameday`."""
    try:
        opt = pd.read_csv('optimal_team_ml.csv')
    except FileNotFoundError:
        return

    if 'Team' not in opt.columns:
        return

    tm = team_game_times(sched, gameday)
    opt['game_time'] = opt['Team'].map(tm)
    opt['game_day'] = opt['game_time'].apply(
        lambda dt: dt.strftime('%a %H:%M') if pd.notna(dt) else '?'
    )

    # Flag players whose only prior stats come from Eurocup (EC→EL transfers)
    # These are in the pool but predictions are low-confidence
    ec_flag = _detect_ec_transfers(opt)

    print(f"\n{'='*60}")
    print(f"  YOUR TEAM — GAME TIMES  (Round {gameday})")
    print(f"{'='*60}")
    print(f"\n  {'':2} {'':1} {'Player':<24}  {'Pos':<4}  {'Team':<5}  {'Game':<11}  {'Pred':>6}")
    print(f"  {'-'*58}")

    for _, p in opt.sort_values('game_day').iterrows():
        cap  = '★' if p.get('is_captain') else ' '
        role = 'S' if p.get('is_starter') else 'B'
        flag = ' ⚡EC' if p['Player'] in ec_flag else ''
        print(f"  {cap} {role} {p['Player']:<24}  {p.get('Position','?'):<4}  "
              f"{p['Team']:<5}  {p['game_day']:<11}  "
              f"{p.get('predicted_fp', 0):>6.1f}{flag}")

    if ec_flag:
        print(f"\n  ⚡EC = transferred from Eurocup — prediction is low-confidence")


def _detect_ec_transfers(opt: pd.DataFrame) -> set[str]:
    """
    Return player names that are in the EL price list but whose recent stats
    are mostly from Eurocup games (rough heuristic via Competition column).
    """
    try:
        feats = pd.read_csv('data/processed/player_games_with_features.csv')
        if 'Competition' not in feats.columns:
            return set()
        players = set(opt['Player'])
        flagged = set()
        for player in players:
            rows = feats[feats['Player'] == player]
            if rows.empty:
                continue
            recent = rows.tail(10)
            ec_share = (recent.get('Competition', pd.Series()) == 'Eurocup').mean()
            if ec_share > 0.7:
                flagged.add(player)
        return flagged
    except Exception:
        return set()


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Fetching EuroLeague schedule…")
    sched = load_el_schedule(args.season)

    gameday = args.round if args.round is not None else detect_round(sched)
    print(f"Round: {gameday}")

    print_round(sched, gameday)

    print(f"\n{'='*60}")
    print("  TEAM → GAME (all EL teams this round)")
    print(f"{'='*60}")
    tm = team_game_times(sched, gameday)
    for code, dt in sorted(tm.items(), key=lambda x: x[1]):
        print(f"  {code:<5}  {dt.strftime('%a %b %d  %H:%M')}")

    print_team_with_schedule(sched, gameday)
    print()