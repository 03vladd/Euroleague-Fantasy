## ADDED Requirements

### Requirement: DNP risk score per Day-2 player
The system SHALL compute a DNP risk score for each player in the current team whose team plays on Day 2 or later within the round, based on their recent minutes history.

DNP rate is defined as: (games with 0 minutes played in last 5 games) / 5.

#### Scenario: Player with no recent DNPs
- **WHEN** a Day-2 starter has played minutes in all of their last 5 games
- **THEN** their DNP risk is displayed as 0% and marked LOW

#### Scenario: Player with one recent DNP
- **WHEN** a Day-2 player missed 1 of their last 5 games (0 minutes)
- **THEN** their DNP risk is displayed as 20% and marked MEDIUM

#### Scenario: Player with multiple recent DNPs
- **WHEN** a Day-2 player missed 2 or more of their last 5 games
- **THEN** their DNP risk is displayed as ≥40% and marked HIGH

#### Scenario: No history available
- **WHEN** a player has fewer than 3 games of history
- **THEN** DNP risk is displayed as "?" and marked UNKNOWN

### Requirement: Ranked risk table output
The system SHALL print a table of Day-2 players sorted descending by DNP risk, showing: role (S/B), player name, team, game time, predicted FP, DNP rate, and risk label.

#### Scenario: Mixed starters and bench in table
- **WHEN** both starters and bench players have Day-2 games
- **THEN** both appear in the table, with starters annotated so users know which DNPs are costly