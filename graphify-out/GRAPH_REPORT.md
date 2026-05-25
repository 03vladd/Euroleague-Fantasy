# Graph Report - Euroleague  (2026-05-26)

## Corpus Check
- 36 files · ~28,367 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 656 nodes · 1005 edges · 53 communities (28 shown, 25 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 83 edges (avg confidence: 0.54)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `565ea04f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]

## God Nodes (most connected - your core abstractions)
1. `EuroLeagueData` - 48 edges
2. `EuroLeagueData` - 38 edges
3. `get_requests()` - 29 edges
4. `GameStats` - 25 edges
5. `BoxScoreData` - 22 edges
6. `BoxScoreData` - 21 edges
7. `PlayerStats` - 20 edges
8. `PlayByPlay` - 20 edges
9. `BoxScoreData` - 19 edges
10. `TeamStats` - 17 edges

## Surprising Connections (you probably didn't know these)
- `Euroleague Fantasy ML Project` --conceptually_related_to--> `Euroleague API Python Package`  [INFERRED]
  README.md → euroleague_api-main/README.md
- `Euroleague Fantasy Competition ML Prediction` --conceptually_related_to--> `Euroleague API Python Package`  [INFERRED]
  README.md → euroleague_api-main/README.md
- `A class for getting the player-level stats and data.      Args:         competit` --rationale_for--> `PlayerStats Class`  [EXTRACTED]
  euroleague_api-main/src/euroleague_api/player_stats.py → euroleague_api-main/docs/euroleague_api/player_stats.md
- `BoxScoreData` --uses--> `EuroLeagueData`  [INFERRED]
  euroleague_api-main/src/euroleague_api/boxscore_data.py → euroleague_api-main/src/euroleague_api/EuroLeagueData.py
- `PlayByPlay` --uses--> `BoxScoreData`  [INFERRED]
  euroleague_api-main/src/euroleague_api/play_by_play_data.py → euroleague_api-main/src/euroleague_api/boxscore_data.py

## Communities (53 total, 25 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (53): bool, EuroLeagueData, A function that returns the game metadata, e.g. gamecodes of a round         in, Base class for collecting Euroleague and Eurocup competition's data.      Args:, A wrapper function for getting game data for all games in a single         round, A wrapper function for getting game data for all games in a single         seaso, A wrapper function with the all game data in a range of seasons          Args:, init function for the EuroLeagueData class.          Args:             competiti (+45 more)

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (18): A class for getting the game related stats and data.      Args:         competit, Get game report data for *all* games in a single season          Args:, Get game report data for *all* games in a range of seasons          Args:, Get game stats data for single game          Args:              season (int): Th, A function that gets the game stats data         of *all* games in a single roun, Get game stats data for *all* games in a single season          Args:, Get game stats data for *all* games in a range of seasons          Args:, A function that gets the "teams comparison" game stats for a single         game (+10 more)

### Community 2 - "Community 2"
Cohesion: 0.33
Nodes (4): objective(), Hyperparameter tuning + feature importance analysis.  Run this once before train, float, Trial

### Community 3 - "Community 3"
Cohesion: 0.13
Nodes (25): BoxScoreData, The players' and team's total stats of a particular game.          Args:, A class for getting box-score data      Args:         competition (str, optional, A function that gets the boxscore quarter data of all games in a         particu, A function that gets the boxscore quarter data of *all* games in a         singl, A function that return the player boxscore stats for all games in a         sing, A function that return the player boxscore stats for all games in a         sing, A helper function that gets the boxscore data of a particular data.          Arg (+17 more)

### Community 4 - "Community 4"
Cohesion: 0.05
Nodes (26): Ancestors (in MRO), Attributes, Class variables, Classes, code:python3 (BASE_URL), code:python3 (V1), code:python3 (V2), code:python3 (V3) (+18 more)

### Community 5 - "Community 5"
Cohesion: 0.20
Nodes (14): A function that gets the play-by-play data of *all* games in a range of, DataFrame, PlayByPlay, int, PlayByPlay, A function that gets the play-by-play data of *all* games in a single         se, A function that gets the play-by-play data of *all* games in a range of, A class for getting the game play-by-play data.      Args:         competition ( (+6 more)

### Community 6 - "Community 6"
Cohesion: 0.16
Nodes (26): BoxScoreData Class, Shot Data Concept, DataFrame, PlayerStats, int, str, A class for getting the player-level stats and data.      Args:         competit, A wrapper function for collecting the leading players in a given         stat ca (+18 more)

### Community 7 - "Community 7"
Cohesion: 0.05
Nodes (27): PlayByPlay, Ancestors (in MRO), Attributes, Class variables, Classes, code:python3 (logger), code:python3 (BASE_URL), code:python3 (V1) (+19 more)

### Community 8 - "Community 8"
Cohesion: 0.20
Nodes (7): calculate_fantasy_points(), calculate_form(), calculate_usage(), Feature Engineering for Fantasy Predictions Builds rolling averages, form indica, Calculate form indicators, Rolling usage trend from per-team-game minutes_rank (computed at df level), Calculate fantasy points based on official rules

### Community 9 - "Community 9"
Cohesion: 0.20
Nodes (9): colsample_bytree, gamma, learning_rate, max_depth, min_child_weight, n_estimators, reg_alpha, reg_lambda (+1 more)

### Community 10 - "Community 10"
Cohesion: 0.29
Nodes (4): looks_like_player_list(), Scrape player prices from EuroLeague Fantasy Challenge.  The site is a Flutter W, Walk a JSON blob and return a flat list of dicts that have     both a name field, Walk a JSON blob and return a flat list of dicts that have     both a name field

### Community 11 - "Community 11"
Cohesion: 0.40
Nodes (6): Euroleague API Python Package, Euroleague Fantasy Competition ML Prediction, Pre-commit Configuration, Python Publish GitHub Workflow, Euroleague API Requirements, Euroleague Fantasy ML Project

### Community 12 - "Community 12"
Cohesion: 0.70
Nodes (4): main(), print_header(), Euroleague Fantasy Optimizer - Main Pipeline Run this to execute the full workfl, run_step()

### Community 21 - "Community 21"
Cohesion: 0.05
Nodes (25): PlayerStats, Ancestors (in MRO), Attributes, Class variables, Classes, code:python3 (BASE_URL), code:python3 (V1), code:python3 (V2) (+17 more)

### Community 22 - "Community 22"
Cohesion: 0.06
Nodes (22): TeamStats, Ancestors (in MRO), Attributes, Class variables, Classes, code:python3 (BASE_URL), code:python3 (V1), code:python3 (V2) (+14 more)

### Community 23 - "Community 23"
Cohesion: 0.06
Nodes (21): Ancestors (in MRO), Attributes, Class variables, Classes, code:python3 (BASE_URL), code:python3 (V1), code:python3 (V2), code:python3 (V3) (+13 more)

### Community 24 - "Community 24"
Cohesion: 0.06
Nodes (21): ShotData, Ancestors (in MRO), Attributes, Class variables, Classes, code:python3 (BASE_URL), code:python3 (V1), code:python3 (V2) (+13 more)

### Community 25 - "Community 25"
Cohesion: 0.07
Nodes (19): Attributes, Class variables, Classes, code:python3 (logger), code:python3 (BASE_URL), code:python3 (V1), code:python3 (V2), code:python3 (V3) (+11 more)

### Community 26 - "Community 26"
Cohesion: 0.07
Nodes (18): Standings, Ancestors (in MRO), Attributes, Class variables, Classes, code:python3 (BASE_URL), code:python3 (V1), code:python3 (V2) (+10 more)

### Community 27 - "Community 27"
Cohesion: 0.11
Nodes (17): Auxiliary functions, Boxscore data, code:bash (pip install euroleague-api), code:python (from euroleague_api.shot_data import ShotData), Documentation, Euroleague API, Euroleague Data class, Example (+9 more)

### Community 28 - "Community 28"
Cohesion: 0.15
Nodes (12): code:python3 (boxscore_data), code:python3 (game_metadata), code:python3 (game_stats), code:python3 (play_by_play_data), code:python3 (player_stats), code:python3 (shot_data), code:python3 (standings), code:python3 (team_stats) (+4 more)

### Community 29 - "Community 29"
Cohesion: 0.18
Nodes (7): get_data_over_collection_of_games, get_requests, raise_error, code:python3 (logger), Functions, Module euroleague_api.utils, Variables

### Community 30 - "Community 30"
Cohesion: 0.22
Nodes (6): Ancestors (in MRO), Attributes, get_game_data, get_season_data_from_game_data, Methods, GameStats

### Community 31 - "Community 31"
Cohesion: 0.25
Nodes (7): 1. Grant of License, 2. License Fee, 3. Restrictions, 4. Warranty & Liability, 5. Termination, Commercial License Agreement, Dual License: GPLv3 or Commercial License

### Community 32 - "Community 32"
Cohesion: 0.33
Nodes (5): DataFrame, str, optimise_team(), backtest.py — Round-by-round backtest of ML fantasy team predictions.  For each, Pick the optimal 10-player team + captain from `pool`.     `pool` must have colu

### Community 33 - "Community 33"
Cohesion: 0.40
Nodes (5): Class variables, code:python3 (BASE_URL), code:python3 (V1), code:python3 (V2), code:python3 (V3)

## Knowledge Gaps
- **114 isolated node(s):** `DataFrame`, `str`, `Trial`, `str`, `n_estimators` (+109 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **25 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EuroLeagueData` connect `Community 0` to `Community 1`, `Community 3`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Why does `EuroLeagueData` connect `Community 0` to `Community 1`, `Community 3`, `Community 5`, `Community 6`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Why does `PlayerStats Class` connect `Community 6` to `Community 0`?**
  _High betweenness centrality (0.019) - this node is a cross-community bridge._
- **Are the 29 inferred relationships involving `EuroLeagueData` (e.g. with `BoxScoreData` and `int`) actually correct?**
  _`EuroLeagueData` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 29 inferred relationships involving `EuroLeagueData` (e.g. with `BoxScoreData` and `GameMetadata`) actually correct?**
  _`EuroLeagueData` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `BoxScoreData` (e.g. with `EuroLeagueData` and `PlayByPlay`) actually correct?**
  _`BoxScoreData` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `DataFrame`, `str`, `backtest.py — Round-by-round backtest of ML fantasy team predictions.  For each` to the rest of the system?**
  _177 weakly-connected nodes found - possible documentation gaps or missing edges._