## Context

The pipeline currently consists of:
- `build_features.py` — engineers rolling features (fp_last_3/5/10, minutes, starter rate, plus/minus, opponent defensive rating, home/away flag, etc.) from the raw game-by-game CSV
- `train_model.py` — trains XGBoost on 27 features, CV MAE 5.67; outputs `models/fantasy_predictor.pkl` and `models/feature_columns.pkl`
- `optimize_team_v2.py` — ILP (PuLP/CBC) that picks 10 players + 6 starters + 1 captain under budget/position/team constraints; `predicted_fp` is the raw model output with no risk adjustment
- `coach_optimizer.py` — standalone script that recommends a coach; not yet wired into the optimizer

Pain points identified through backtest analysis:
- Model may pick injury-prone players because DNP history is not a prediction signal
- Captain is always the highest predicted-FP player even if they play on Day-1 (no sub flexibility)
- Coach is a separate manual step after running the optimizer
- Home/away performance split is not a per-player signal (only a binary flag)

## Goals / Non-Goals

**Goals:**
- Add `dnp_rate_last5` and `dnp_rate_last10` to the feature set so the model learns DNP risk
- Add `home_advantage` (cumulative home-vs-away FP delta) as a contextual feature
- Apply a configurable DNP-probability discount to `predicted_fp` before the ILP so risk-prone players are deprioritised without hardcoding exclusions
- Emit a coach recommendation alongside the team in `optimize_team_v2.py`
- When `--round` is given and the round has ≥ 2 game days, constrain the captain variable to Day-2 players

**Non-Goals:**
- Live injury feed integration (no external API for real-time news)
- Price-change modelling (dynamic capital gains — acknowledged gap, separate effort)
- Changing the ILP solver or switching to a heuristic search
- Building a UI or web service

## Decisions

### D1 — DNP discount formula: multiplicative linear

`adj_fp = predicted_fp × max(0, 1 − DNP_DISCOUNT × dnp_rate_last5)`

`DNP_DISCOUNT = 0.5` means a player who DNP'd in all 5 recent games has their predicted FP halved; a player who never DNP'd is unaffected. The constant is tunable via a top-level config var.

**Alternative considered:** Additive penalty (`adj_fp = predicted_fp − k × dnp_rate`). Rejected because the penalty should scale with the player's FP level — a 20-pt star losing 2pts to DNP risk is different from a 6-pt bench filler losing 2pts.

**Alternative considered:** Hard exclusion (`if dnp_rate_last5 > 0.6: skip player`). Rejected — too blunt; a star returning from injury may have high DNP rate but strong predicted FP, and the LP budget constraint naturally handles the trade-off.

### D2 — Home advantage: cumulative expanding mean, not rolling window

`home_advantage = expanding_mean(fp | home) − expanding_mean(fp | away)` per player, using only prior games (shift not needed because we compute within `groupby('Player')` and the feature is assigned to the current row before training).

**Alternative considered:** Rolling 5-game home/away split. Rejected — most players have only 2-3 home games in any 5-game window, so rolling std would be huge; expanding mean is more stable.

### D3 — Coach price scraping: extend existing Playwright flow in `scrape_prices.py`

The existing scraper opens the EuroLeague Fantasy market in a headed Playwright browser (required — Flutter WebAssembly won't render headlessly) and scrapes player prices by position tab. Coaches have their own market tab on the same site.

**Decision:** Add coach scraping as a second pass in the same browser session inside `scrape_prices.py`. After scraping player prices, navigate to the Coach market tab, extract each coach row (team code + price), and save to `coach_prices.csv`. This keeps a single scraping script and avoids a second browser launch.

**Alternative considered:** Separate `scrape_coach_prices.py`. Rejected — two browser launches for the same session is wasteful; the user already logs in manually once per scrape run.

### D4 — Coach co-optimised in ILP, not post-hoc

The coach eats into the same 100-credit budget as the 10 players. Post-hoc selection (pick the best coach after fixing the player team) would under-optimise because a cheaper coach may free enough budget to upgrade a player worth more EV than the coach FP gap.

**Decision:** Extend the ILP with binary variables `coach_var[j]` for each coach playing in the round. Budget constraint becomes `Σ player_prices + Σ coach_prices × coach_var ≤ 100`. Objective adds `Σ expected_coach_fp[j] × coach_var[j]`. Constraint `Σ coach_var[j] == 1`.

Expected coach FP computation (inline, same formula as `coach_optimizer.py`): cumulative win rate + last-5 win rate + home bonus → win probability → expected FP using season margin distribution.

### D5 — Schedule-aware captain: LP hard constraint, not soft penalty

Add a conditional constraint:
```python
if d2_indices:
    for i in df.index:
        if i not in d2_indices:
            prob += captain_vars[i] == 0
```

This is cleaner than a penalty weight (which might produce fractional-seeming LP solutions and is hard to calibrate). If no D2 players are in the predicted squad, the constraint is omitted and captain selection falls back to unconstrained.

**Alternative considered:** Soft penalty (reduce captain bonus for D1 players). Rejected — the captain bonus is already encoded in the LP objective; adding a penalty parameter makes the objective function semantics opaque.

### D6 — Feature sync across scripts

`backtest.py` and `backtest_subs.py` both define `FEATURE_COLS` as a hardcoded list. After adding new features, those lists must match. **Decision:** Load feature columns from `models/feature_columns.pkl` in both backtest scripts (already the authoritative source saved by `train_model.py`) rather than keeping a third copy.

## Risks / Trade-offs

- **DNP_DISCOUNT calibration** → The 0.5 default is a starting point. If the season's data shows that even high-DNP players who play score well, the discount may over-penalise. Mitigation: expose as a config constant at the top of the optimizer; document in CLAUDE.md for easy tuning next season.

- **home_advantage NaN for new players** → Players with only home or only away games will produce NaN from the `away_mean` or `home_mean`. Mitigation: fill NaN with 0 (no known advantage either way).

- **Coach inline logic divergence** → If `coach_optimizer.py` is updated, the inlined logic in `optimize_team_v2.py` won't automatically sync. Mitigation: add a comment pointing to `coach_optimizer.py` as the canonical source; mark for extraction into a shared util if divergence becomes a problem.
- **Coach market UI changes** → The EuroLeague Fantasy site may reorganise the Coach tab between seasons. Mitigation: Playwright selector is isolated to the coach scraping block; `scrape_prices.py` fails loudly (not silently) if the coach tab is not found, so the user knows to update the selector.

- **Schedule API latency** → The captain D2 constraint requires a live schedule fetch (euroleague_api); this adds ~1-2s to optimizer runtime. Mitigation: already handled by `schedule_helper.py` which caches the schedule object in-process.

## Migration Plan

1. Extend `scrape_prices.py` with coach tab scraping — run it to populate `coach_prices.csv`
2. Run `python3 build_features.py` — regenerates feature CSVs with new DNP and home-advantage columns
3. Run `python3 train_model.py` — retrains model on new feature set, updates `models/feature_columns.pkl`
4. `backtest.py` and `backtest_subs.py` updated to load `FEATURE_COLS` from pkl
5. `optimize_team_v2.py` updated with DNP discount, coach ILP extension, and D2 captain constraint
6. Rollback: restore previous pkl files from git; revert feature-engineering and optimizer edits; delete `coach_prices.csv`

## Open Questions

- Should the DNP discount also apply to the backtest (so we measure its actual lift in CV)? Currently it's only in the optimizer. A follow-up backtest run with the new model will implicitly include DNP features; the discount is an optimizer-level decision not a model-level one, so the backtest wouldn't capture it unless we add it to `backtest.py` as well.
- Should `home_advantage` be split into current-game context (`is_home` for next game) vs player-level tendency? The `Home` flag already exists as a feature; `home_advantage` adds the player-level tendency on top.