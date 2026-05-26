## ADDED Requirements

### Requirement: DNP rate rolling features computed per player
`build_features.py` SHALL compute two new columns inside `calculate_rolling_stats`:
- `dnp_rate_last5`: fraction of the player's last 5 games where `Minutes == 0`, using `rolling(5, min_periods=1).mean()`
- `dnp_rate_last10`: same over the last 10 games

Both columns SHALL be computed using only past game data (the rolling window is applied in chronological order within the player group, so no future leakage occurs).

Both columns SHALL be included in the `prediction_features` export and in `train_model.py`'s `feature_cols` list.

#### Scenario: Player with recent DNPs has non-zero rate
- **WHEN** a player has 0 minutes in 2 of their last 5 games
- **THEN** `dnp_rate_last5` equals 0.4 for their next row

#### Scenario: Player who always plays has zero rate
- **WHEN** a player has > 0 minutes in every game
- **THEN** `dnp_rate_last5` equals 0.0

#### Scenario: Early-season player with fewer than 5 games
- **WHEN** a player has played only 3 games and none were DNPs
- **THEN** `dnp_rate_last5` equals 0.0 (min_periods=1 fills with available data)

#### Scenario: New features propagated to prediction CSV
- **WHEN** `build_features.py` runs to completion
- **THEN** `data/processed/prediction_features.csv` contains columns `dnp_rate_last5` and `dnp_rate_last10`

#### Scenario: New features present in model feature set
- **WHEN** `train_model.py` runs after `build_features.py`
- **THEN** `models/feature_columns.pkl` contains `dnp_rate_last5` and `dnp_rate_last10`