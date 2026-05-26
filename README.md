# EuroLeague Fantasy — ML Optimizer

End-to-end machine learning system for the [EuroLeague Fantasy Challenge](https://euroleaguefantasy.euroleaguebasketball.net). Predicts player fantasy points using XGBoost, then selects the optimal 10-player + coach squad using Integer Linear Programming under real competition constraints (budget cap, position requirements, per-team limits).

Backtested across the full 2025-26 EuroLeague regular season (38 rounds): **+28% above a rolling-average naive baseline**, achieving 58% of a perfect-information oracle.

---

## Results

| Strategy | Avg FP / round | Season total | vs Naive |
|---|---|---|---|
| **ML optimizer** (this system) | **131.6** | **4,345** | — |
| Naive baseline (fp_last_3 heuristic) | 102.4 | 3,379 | −23% |
| Oracle (LP on actual FP, perfect info) | 225.8 | 7,451 | +120% |

- ML beats the naive baseline in **23 of 33** evaluated rounds
- ML achieves **58.3%** of the perfect-information oracle score
- Walk-forward validation: model only ever trained on past rounds, never future data

Interactive dashboard → open `dashboard.html` in any browser (no server required).

---

## Pipeline

```
prepare_data.py       # fetch game-by-game stats from EuroLeague API
    ↓
build_features.py     # engineer 35 predictive features (rolling form, rest, matchup, consistency)
    ↓
tune_model.py         # Optuna hyperparameter search (optional, saves best_params.json)
    ↓
train_model.py        # XGBoost regression, time-series CV, generate next_round_predictions.csv
    ↓
optimize_team_v2.py   # ILP team selection (PuLP/CBC), co-optimised coach, D2 captain constraint
    ↓
transfer_optimizer.py # suggest 1–3 transfers from current squad (reads my_team.json)
    ↓
backtest.py           # walk-forward evaluation over all regular-season rounds
    ↓
dashboard.py          # generate interactive Plotly dashboard → dashboard.html
```

Price data is collected separately before each round:
```
scrape_prices.py      # Playwright browser automation → euroleague_prices.csv + coach_prices.csv
```

---

## Features (35 total)

| Category | Features |
|---|---|
| **Recent form** | fp_last_3/5/10, fp_std_3/5/10 |
| **Minutes / usage** | minutes_last_3/5/10, minutes_rank, usage_trend |
| **Role stability** | starter_rate_3/5 |
| **Team quality** | plusminus_last_3/5, team_won |
| **Form signals** | form_trend, hot_streak, points_share |
| **Matchup** | opp_defensive_rating, opp_def_pos_rating (G/F/C split) |
| **Context** | Home, games_played |
| **Consistency** | fp_cv_5, fp_cv_10 (coefficient of variation) |
| **Rest / fatigue** | days_rest, short_rest (inter-round gap ≤ 3 days) |
| **DNP risk** | dnp_rate_last5, dnp_rate_last10 |
| **Home tendency** | home_advantage (expanding home vs away FP differential) |
| **Box score** | Points, TotalRebounds, Assistances, Valuation |

All rolling features use a strict left-shift (`shift(1)`) to prevent future leakage.

---

## Optimization formulation

The team selection is cast as a **Mixed Integer Linear Program** solved with PuLP/CBC:

```
maximize  Σᵢ [ fp_adj[i] · (0.5·p[i] + 0.5·s[i]) + cap_score[i] · c[i] ]
        + Σⱼ exp_fp_coach[j] · k[j]

subject to
  Σ p[i]                    = 10     (roster size)
  Σ s[i]                    = 6      (6 starters at 100%, 4 bench at 50%)
  Σ c[i]                    = 1      (one captain, scores 2×)
  Σ p[i]·price[i]
    + Σ k[j]·coach_price[j] ≤ 100   (shared budget cap)
  Σ p[i] for team t         ≤ 6      (per-team limit)
  p[i] ∈ {G:4, F:4, C:2}            (position requirements)
  s[i] ≤ p[i],  c[i] ≤ s[i]        (role hierarchy)
  Σ k[j]                    = 1      (one coach)
  c[i] = 0  for Day-1 players        (schedule-aware captain, when --round is given)
```

Where:
- `fp_adj[i]` = `predicted_fp × max(0, 1 − 0.5 × dnp_rate_last5)` — DNP risk discount
- `cap_score[i]` = `fp_adj[i] − 0.3 × fp_std_5[i]` — captain reliability penalty (prefers consistent players over volatile ones with similar predicted FP)
- `exp_fp_coach[j]` = modelled from team win-rate, recent form, and home/away status

---

## Technical design choices

**Time-series cross-validation** — `TimeSeriesSplit(n_splits=5)` ensures training always precedes validation in time. No player's future performance ever contaminates their past feature window.

**Walk-forward backtest** — For each round R, the model is retrained from scratch on rounds 1..R−1, then evaluated on R. This mirrors real deployment: you only know what happened before the current round.

**DNP risk discount** — Players with recent DNP history get their predicted FP discounted before entering the ILP, reducing the chance of wasting a budget slot on someone who might not play.

**Captain reliability penalty** — The LP uses a consistency-adjusted captain score, penalising high-variance players for the captaincy even if their predicted FP is marginally higher, avoiding volatile players who might DNP or have an off-night.

**Position-specific defensive rating** — Opponents are rated separately by how many FP they allow to Guards, Forwards, and Centers (rolling 5-game mean, 1-round lag to prevent leakage), rather than a single team-level defensive number.

**Coach co-optimisation** — Coach selection shares the same 100-credit budget and is solved jointly with player selection in a single ILP pass.

**Schedule-aware captain** — When `--round` is provided, the optimizer fetches the actual game schedule and constrains the captain to Day-2 players only (so you can make the decision after seeing Day-1 results without sacrificing captain points).

---

## Setup

```bash
pip install -r requirements.txt
playwright install chromium   # for price scraping only
```

**Run order (start of new season):**
```bash
python3 prepare_data.py       # ~6h due to API rate limits
python3 build_features.py
python3 tune_model.py         # optional — Optuna hyperparameter search
python3 train_model.py
python3 scrape_prices.py      # log in → browse market → close window
python3 optimize_team_v2.py --round <N>
```

**Each round:**
```bash
python3 scrape_prices.py      # update prices before transfer window closes
python3 train_model.py        # retrain on latest data
python3 optimize_team_v2.py --round <N>
python3 transfer_optimizer.py --transfers 2   # optional: suggest swaps from current squad
```

**Analysis:**
```bash
python3 backtest.py     # walk-forward evaluation → backtest_results.csv
python3 dashboard.py    # → dashboard.html (open in any browser)
```

---

## Repository structure

```
├── prepare_data.py          # Data collection (EuroLeague API)
├── build_features.py        # Feature engineering (35 features)
├── tune_model.py            # Hyperparameter optimisation (Optuna)
├── train_model.py           # XGBoost training + predictions
├── optimize_team_v2.py      # ILP team optimizer (main entry point)
├── transfer_optimizer.py    # Suggest transfers from current squad
├── backtest.py              # Walk-forward season backtest
├── backtest_subs.py         # Mid-round substitution backtest
├── schedule_helper.py       # Game schedule + transfer window info
├── scrape_prices.py         # Playwright price scraper
├── dashboard.py             # Interactive Plotly dashboard
├── my_team.json             # Current squad (auto-written by optimizer)
├── requirements.txt
└── openspec/                # Feature specs and design documents
```

Data files (gitignored): `data/`, `models/*.pkl`, `*.csv` in root.

---

## Tech stack

| Layer | Technology |
|---|---|
| Data collection | `euroleague_api`, Playwright browser automation |
| Feature engineering | Pandas, NumPy |
| ML model | XGBoost (gradient boosting regression) |
| Hyperparameter tuning | Optuna (Bayesian optimisation) |
| Team optimisation | PuLP + CBC (MILP solver) |
| Visualisation | Plotly |
| Validation | scikit-learn `TimeSeriesSplit` |