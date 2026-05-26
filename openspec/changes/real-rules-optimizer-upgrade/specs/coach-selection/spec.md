## ADDED Requirements

### Requirement: Coach co-optimised in the ILP under the shared 100-credit budget
The coach selection SHALL be part of the same ILP as the 10 player picks. The ILP SHALL be extended with binary variables `coach_var[j]` for each coach whose team plays in the target round.

**Budget constraint (extended):**
```
Σ_i (player_price[i] × player_var[i]) + Σ_j (coach_price[j] × coach_var[j]) ≤ BUDGET
```

**Objective (extended):**
```
maximise: [existing player terms] + Σ_j (expected_coach_fp[j] × coach_var[j])
```

**Coach selection constraint:**
```
Σ_j coach_var[j] == 1
```

Expected coach FP SHALL be computed for each eligible coach using:
1. Cumulative season win rate + last-5 win rate (each weighted 0.5)
2. Home-court bonus (+0.08 to win probability for home team)
3. Win probability → expected FP using the official coach scoring table and observed season margin distribution

Coach data SHALL be loaded from `coach_prices.csv` (team + price). If the file is absent, the optimizer SHALL print a warning, exclude coaches from the ILP, and note that the budget is entirely available for players.

#### Scenario: Coach included when coach_prices.csv is present
- **WHEN** `coach_prices.csv` exists and `--round` resolves to a valid round
- **THEN** the ILP solution includes exactly one coach; the coach's price is deducted from the 100-credit budget

#### Scenario: Cheaper coach preferred when budget is tight
- **WHEN** selecting the highest expected-FP coach would push the budget over 100 credits
- **THEN** the ILP selects a lower-priced coach (or adjusts the player selection) to stay within budget

#### Scenario: Coach output shown in optimizer results
- **WHEN** the ILP finds a feasible solution with a coach
- **THEN** output includes a COACH section with: team, opponent, home/away, win%, expected coach FP, price — and a "★ Coach pick" line

#### Scenario: Coach absent when price file missing
- **WHEN** `coach_prices.csv` does not exist
- **THEN** optimizer runs player selection without coach variable, prints "coach_prices.csv not found — run scrape_prices.py to enable coach co-optimisation", and uses the full 100 credits for players

### Requirement: Coach expected FP loaded from historical game data
The expected coach FP computation SHALL use `data/raw/all_player_game_stats_2025.csv` (team totals rows where `Player == 'Total'`) to derive win rates and margins — the same source used by `coach_optimizer.py`. No additional API calls are made at optimizer runtime.

#### Scenario: Expected FP computed without extra API calls
- **WHEN** optimizer runs
- **THEN** coach FP estimates are derived entirely from the local raw CSV, not from live API requests