## ADDED Requirements

### Requirement: Captain constrained to Day-2 players when schedule is multi-day
When `--round` is supplied and the round's schedule contains at least two distinct game days, `optimize_team_v2.py` SHALL add a constraint to the ILP restricting the captain variable to players whose team plays on Day-2 or later.

Formally, for each player `i` whose team plays on Day-1 (the earliest game day in the round):
```
captain_var[i] == 0
```

This constraint SHALL only be added when:
1. `--round` is provided (schedule data available)
2. The schedule shows ≥ 2 distinct game days
3. At least one selected player's team plays on Day-2 or later (if no D2 players are in the squad, the constraint is omitted and captain selection is unconstrained)

The rationale printed in the optimizer output SHALL explain: "Captain restricted to Day-2 players — you can confirm or change captain after Day-1 results."

#### Scenario: Multi-day round restricts captain to Day-2 player
- **WHEN** `--round 15` is passed and round 15 has games on Thursday and Friday
- **THEN** the captain is a player whose team plays on Friday

#### Scenario: Single-day round leaves captain unconstrained
- **WHEN** `--round` is passed but all games are on the same day
- **THEN** no Day-2 captain constraint is added and captain is the highest adj_fp player

#### Scenario: No Day-2 player in squad falls back to unconstrained
- **WHEN** all 10 selected players happen to play on Day-1
- **THEN** the constraint is skipped, captain is unconstrained, and a warning is printed: "All squad players play on Day-1 — captain constraint not applied"

#### Scenario: No --round flag leaves captain unconstrained
- **WHEN** optimizer is run without `--round`
- **THEN** no Day-2 captain constraint is applied