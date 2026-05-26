# Claude Code — Euroleague Fantasy Project

## End-of-session checklist (do this before signing off)

1. **Update knowledge graph** (no API key needed):
   ```
   graphify update .
   ```

2. **Update Obsidian vault note** at `/mnt/c/Users/vasiu/UBB/The Vault/Projects/Euroleague Fantasy.md`:
   - Updated graph stats (nodes/edges/communities from GRAPH_REPORT.md)
   - Latest optimal team (from `optimal_team_ml.csv`)
   - Any new backtest results
   - Pipeline status / pending data fetches

3. **Commit and push** all changed source files (never commit):
   - `.fantasy_creds.json` — credentials
   - `data/` — large API data
   - `models/*.pkl` — binary model files
   - `*.csv` in root — generated outputs

## Season naming

EuroLeague API uses **start year** as season code:
- `season=2025` → 2025-26 EuroLeague (current, just ended May 2026)
- `season=2024` → 2024-25 (prior season, file: `all_player_game_stats_2024.csv`)

## Pipeline run order

```
python3 prepare_data.py      # ~6h due to API rate limits (0.35s/request)
python3 build_features.py
python3 train_model.py
python3 optimize_team_v2.py
python3 backtest.py          # optional, uses existing features file
```

## Price scraping

Run `python3 scrape_prices.py` — requires WSLg display (Windows 11 with WSL2).
Browser opens headed; log in → navigate to Transfer/Market → browse all positions → navigate to Coach tab → browse all coaches → close window.
Outputs: `euroleague_prices.csv` (players) + `coach_prices.csv` (coaches, needed for coach co-optimisation).
Both files are gitignored. Run before `optimize_team_v2.py` each round.
Credentials stored in `.fantasy_creds.json` (gitignored).

Note: scraper adds a 2s sleep after page load and 1s after cookie dismiss to let the Flutter app initialise and avoid triggering rate limits.

## Schedule / transfer window

```
python3 schedule_helper.py              # auto-detect next round from today's date
python3 schedule_helper.py --round 5   # specific round
python3 optimize_team_v2.py --round 5  # optimizer + game times in one shot
```

- EuroLeague schedule only (Eurocup games irrelevant for team selection)
- EC→EL transfers are included in the player pool but flagged with ⚡EC — predictions are low-confidence (weaker prior competition, new team fit unknown)
- Transfer deadline = first tipoff of the round (shown automatically)

## Key fixes applied (don't revert)

- `euroleague_api/utils.py`: `time.sleep(0.35)` is in `finally` block — applies to every request including errors, prevents TCP rate limiting
- `tune_model.py`: SHAP monkey-patch for XGBoost 3.x (`base_score` bracket format)
- `train_model.py`: final model uses `**BEST_PARAMS` from `models/best_params.json`
- `scrape_prices.py`: `headless=False` required — Flutter WebAssembly won't render headlessly