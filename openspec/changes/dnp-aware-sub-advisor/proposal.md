## Why

Backtest analysis (backtest_subs.py) proved that knowing which players will miss Day-2 games (DNPs) before tipoff is worth **+3.4 pts/round** on average — roughly 1 in 3 rounds has an actionable substitution. Currently there is no tool to surface this risk or apply the sub; users must track injury news manually and re-run the optimizer from scratch.

## What Changes

- New script `dnp_advisor.py` that reads the current team from `optimal_team_ml.csv` and the EL schedule, then:
  - Splits players into Day-1 (already played / locked) and Day-2+ (still to play)
  - Computes a DNP-risk score per Day-2 player from their recent minutes history
  - Prints a ranked risk table so the user knows which starters to watch
  - Accepts `--dnp PLAYER1,PLAYER2` to immediately compute and print the optimal reassigned lineup (promote best bench → starter, reassign captain by predicted FP)

## Capabilities

### New Capabilities
- `dnp-risk-table`: For each Day-2 player in the current team, compute DNP rate over last 5 games and display ranked risk alongside their starter/bench role
- `lineup-reassignment`: Given a confirmed list of DNP players, compute the optimal updated lineup (starter set + captain) using predicted FP from optimal_team_ml.csv

### Modified Capabilities
<!-- none -->

## Impact

- New file: `dnp_advisor.py`
- Reads: `optimal_team_ml.csv`, `data/processed/player_games_with_features.csv`, EL schedule API
- No changes to existing pipeline scripts
- Dependency: `schedule_helper.py` (reuses `load_el_schedule`, `team_game_times`)