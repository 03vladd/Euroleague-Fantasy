## ADDED Requirements

### Requirement: Per-player home-advantage feature computed in feature engineering
`build_features.py` SHALL compute `home_advantage` for each player row as the difference between the player's cumulative mean FP in home games and cumulative mean FP in away games, using only games prior to the current row (expanding window, no leakage).

Formula (per player group, chronological order):
```
home_advantage = expanding_mean(fp where Home==1) − expanding_mean(fp where Home==0)
```

When a player has no home games yet or no away games yet (making one side NaN), `home_advantage` SHALL be filled with `0.0`.

`home_advantage` SHALL be included in the `prediction_features` export and in `train_model.py`'s `feature_cols` list.

#### Scenario: Player who performs better at home
- **WHEN** a player averages 18 FP at home and 12 FP away (cumulative to date)
- **THEN** `home_advantage` equals 6.0 on their next home or away game row

#### Scenario: Player with no away games yet
- **WHEN** a player has played 4 games all at home
- **THEN** `home_advantage` equals 0.0 (NaN from missing away side filled to 0)

#### Scenario: Feature present in prediction CSV
- **WHEN** `build_features.py` runs to completion
- **THEN** `data/processed/prediction_features.csv` contains column `home_advantage` with no NaN values