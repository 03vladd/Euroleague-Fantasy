## 1. Coach Price Scraper

- [x] 1.1 Inspect EuroLeague Fantasy Coach market tab in the Playwright session to identify CSS selectors for coach rows (name, team, price)
- [x] 1.2 Extend `scrape_prices.py` to navigate to the Coach market tab after player prices are collected
- [x] 1.3 Extract each coach row and build a list of `{team, coach_name, price}` dicts
- [x] 1.4 Write `coach_prices.csv` to the project root; raise a clear error if the coach tab selector is not found
- [x] 1.5 Update CLAUDE.md price scraping section to note that `scrape_prices.py` now also outputs `coach_prices.csv`

## 2. Feature Engineering — DNP Risk

- [x] 2.1 Add `dnp` binary column (`Minutes == 0`) inside `calculate_rolling_stats` in `build_features.py`
- [x] 2.2 Compute `dnp_rate_last5` and `dnp_rate_last10` rolling means within the same function
- [x] 2.3 Add `dnp_rate_last5` and `dnp_rate_last10` to the `prediction_features` column list in Step 6 of `build_features.py`

## 3. Feature Engineering — Home Advantage

- [x] 3.1 Add `home_advantage` computation per player group in `build_features.py`: expanding mean of home FP minus expanding mean of away FP, filled with 0 where NaN
- [x] 3.2 Add `home_advantage` to the `prediction_features` column list in Step 6 of `build_features.py`

## 4. Model Retrain

- [x] 4.1 Add `dnp_rate_last5`, `dnp_rate_last10`, and `home_advantage` to `feature_cols` in `train_model.py`
- [x] 4.2 Run `python3 build_features.py` and verify new columns appear in `data/processed/prediction_features.csv`
- [x] 4.3 Run `python3 train_model.py` and verify `models/feature_columns.pkl` contains the three new columns; note new CV MAE

## 5. Backtest Feature Sync

- [x] 5.1 Update `backtest.py` to load `FEATURE_COLS` from `models/feature_columns.pkl` instead of the hardcoded list
- [x] 5.2 Update `backtest_subs.py` to load `FEATURE_COLS` from `models/feature_columns.pkl` instead of the hardcoded list
- [x] 5.3 Run `python3 backtest.py` and confirm results are consistent (similar ML avg FP/round)

## 6. Optimizer — DNP Risk Penalty

- [x] 6.1 Add `DNP_DISCOUNT = 0.5` constant at the top of `optimize_team_v2.py`
- [x] 6.2 After loading predictions, compute `adj_predicted_fp` column applying the discount formula; fall back gracefully if `dnp_rate_last5` is absent
- [x] 6.3 Update `build_prob` to use `adj_predicted_fp` in the LP objective instead of `predicted_fp`
- [x] 6.4 Add `AdjFP` column to the printed team table and print the DNP risk section below the team

## 7. Optimizer — Coach Co-optimisation

- [x] 7.1 Add coach data loading block: read `data/raw/all_player_game_stats_2025.csv` team totals, compute win rates and margins, build `coach_df` with expected FP per team
- [x] 7.2 Load `coach_prices.csv`; merge with `coach_df` on `team`; print warning and skip coach block if file absent
- [x] 7.3 Extend `build_prob` to add binary `coach_vars`, the `Σ coach_var == 1` constraint, the extended budget constraint, and the coach FP objective term
- [x] 7.4 Extract and print the selected coach in the results section (team, opponent, H/A, win%, exp FP, price)

## 8. Optimizer — Schedule-Aware Captain

- [x] 8.1 In the `--round` schedule block, after fetching game times, determine `d1_teams` (teams with the earliest game day) and `d2_player_indices` (selected players from non-D1 teams)
- [x] 8.2 Add LP constraints `captain_var[i] == 0` for all D1 players, conditional on `d2_player_indices` being non-empty
- [x] 8.3 Print a note in the output explaining why the captain is a D2 player (or print a warning if constraint was skipped)

## 9. Verification

- [x] 9.1 Run `python3 optimize_team_v2.py --round <next_round>` end-to-end; verify team + coach output, DNP risk table, and D2 captain
- [x] 9.2 Confirm budget total (players + coach) ≤ 100 in the output
- [x] 9.3 Commit all changed source files; update CLAUDE.md pipeline section