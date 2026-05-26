## Why

The current ML pipeline and optimizer are not using the full information available under the real EuroLeague Fantasy competition rules. The backtest shows a 13.8 pt/round gap between the raw ML lineup and what an informed manager can achieve through retroactive reassignments, plus an additional 21.9 pts/round available from the coach pick — neither of which the optimizer currently models. With the regular season concluded and VIP Final Four tickets on the line for top managers, every edge matters.

## What Changes

- **New feature: `dnp_rate_last5` / `dnp_rate_last10`** — per-player DNP rate (0-minute games) over last 5 and 10 games, added to `build_features.py` and `train_model.py` feature set. Allows the model to self-penalise injury-prone players.
- **New feature: `home_advantage`** — per-player cumulative home-vs-away FP differential, backward-looking to avoid leakage. Added to `build_features.py` and feature set.
- **DNP risk discount in optimizer** — before solving the ILP, scale each player's `predicted_fp` down by their DNP probability so the LP naturally avoids high-risk selections.
- **Coach price scraper** — extend `scrape_prices.py` (or add `scrape_coach_prices.py`) to also scrape each EL coach's name, team, and current price from the EuroLeague Fantasy market, saving to `coach_prices.csv`.
- **Coach co-optimised in ILP** — the coach is part of the 100-credit shared budget; extend the ILP in `optimize_team_v2.py` to pick the best-value coach (expected FP per credit) under the same budget constraint as players.
- **Schedule-aware captain constraint** — when `--round` is supplied and a multi-day schedule is detected, restrict the captain variable in the LP to Day-2 players so the manager retains the option to confirm or change captain after seeing Day-1 results.
- **Model retrain** — run `build_features.py` → `train_model.py` after feature additions to incorporate new signals.

## Capabilities

### New Capabilities

- `dnp-risk-features`: Compute and expose per-player DNP rate features (`dnp_rate_last5`, `dnp_rate_last10`) in the feature-engineering pipeline and model.
- `home-advantage-feature`: Compute per-player cumulative home-vs-away FP differential as a rolling feature.
- `dnp-risk-optimizer-penalty`: Apply a DNP-probability discount to `predicted_fp` before solving the ILP so risk-prone players are naturally deprioritised.
- `coach-price-scraper`: Scrape each coach's current price from the EuroLeague Fantasy website and save to `coach_prices.csv`, integrated into the existing Playwright scraping flow.
- `coach-selection`: Co-optimise the coach inside the same ILP as the 10 players, sharing the 100-credit budget; expected coach FP computed from win probability model; coach with best value (FP per credit within budget) is selected.
- `schedule-aware-captain`: When round schedule is available and the round has multiple game days, constrain the captain LP variable to Day-2 players.

### Modified Capabilities

- `dnp-risk-table`: The existing spec in `openspec/changes/dnp-aware-sub-advisor/specs/dnp-risk-table/spec.md` defined a standalone display table. This change supersedes it by embedding DNP risk directly into the feature pipeline and optimizer — the table output remains but is now derived from the feature data rather than a separate computation.

## Impact

- `build_features.py` — new rolling columns added inside `calculate_rolling_stats`; new columns propagated through `prediction_features` export
- `train_model.py` — two new columns added to `feature_cols`; model retrained and pickled
- `optimize_team_v2.py` — DNP discount applied pre-LP; LP extended with coach binary vars + budget constraint; captain D2 constraint added conditionally
- `scrape_prices.py` (or new `scrape_coach_prices.py`) — adds coach market scraping; outputs `coach_prices.csv`
- `backtest.py` / `backtest_subs.py` — `FEATURE_COLS` list must be kept in sync with new columns (or loaded from `models/feature_columns.pkl`)
- New output file: `coach_prices.csv` (gitignored, updated each round alongside `euroleague_prices.csv`)
- No new Python dependencies (Playwright already used for player price scraping)