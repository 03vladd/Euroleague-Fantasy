# Graph Report - .  (2026-05-26)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 216 nodes · 410 edges · 21 communities (15 shown, 6 thin omitted)
- Extraction: 89% EXTRACTED · 11% INFERRED · 0% AMBIGUOUS · INFERRED: 47 edges (avg confidence: 0.57)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `86e34cc3`
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

## God Nodes (most connected - your core abstractions)
1. `EuroLeagueData` - 43 edges
2. `get_requests()` - 23 edges
3. `BoxScoreData` - 22 edges
4. `GameStats` - 14 edges
5. `PlayerStats Class` - 14 edges
6. `raise_error()` - 13 edges
7. `Euroleague API README` - 12 edges
8. `EuroLeagueData Base Class` - 11 edges
9. `int` - 10 edges
10. `DataFrame` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Euroleague Fantasy ML Project` --conceptually_related_to--> `Euroleague API Python Package`  [INFERRED]
  README.md → euroleague_api-main/README.md
- `Euroleague Fantasy Competition ML Prediction` --conceptually_related_to--> `Euroleague API Python Package`  [INFERRED]
  README.md → euroleague_api-main/README.md
- `str` --uses--> `EuroLeagueData`  [INFERRED]
  euroleague_api-main/src/euroleague_api/game_stats.py → euroleague_api-main/src/euroleague_api/EuroLeagueData.py
- `A class for getting the player-level stats and data.      Args:         competit` --rationale_for--> `PlayerStats Class`  [EXTRACTED]
  euroleague_api-main/src/euroleague_api/player_stats.py → euroleague_api-main/docs/euroleague_api/player_stats.md
- `BoxScoreData` --uses--> `EuroLeagueData`  [INFERRED]
  euroleague_api-main/src/euroleague_api/boxscore_data.py → euroleague_api-main/src/euroleague_api/EuroLeagueData.py

## Communities (21 total, 6 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.10
Nodes (26): EuroLeagueData, Base class for collecting Euroleague and Eurocup competition's data.      Args:, init function for the EuroLeagueData class.          Args:             competiti, GameMetadata, Retrieves game metadata for a given gamecode, such as stadium,         capacity, A function to retrieve game metadata for all games in a single season., A class for getting the game related metadata, such as     stadum, capacity and, A function that gets the metadata of *all* games in a range of seasons (+18 more)

### Community 1 - "Community 1"
Cohesion: 0.18
Nodes (13): GameStats, A class for getting the game related stats and data.      Args:         competit, Get game report data for *all* games in a single season          Args:, Get game stats data for single game          Args:              season (int): Th, Get game stats data for *all* games in a single season          Args:, Get game stats data for *all* games in a range of seasons          Args:, A function that gets the "teams comparison" game stats for a single         game, A function that gets the "teams comparison" game stats         of *all* games in (+5 more)

### Community 2 - "Community 2"
Cohesion: 0.15
Nodes (14): DataFrame, int, str, A function that gets the team stats for all seasons          Args:, A class for getting team-level stats and data.      Args:         competition (s, A function that returns the teams' stats in a single season          Args:, A function that returns the teams' stats in a range of seasons          Args:, In this function we derive team advanced stats from a single game         that a (+6 more)

### Community 3 - "Community 3"
Cohesion: 0.19
Nodes (14): bool, BoxScoreData, The players' and team's total stats of a particular game.          Args:, A class for getting box-score data      Args:         competition (str, optional, A function that gets the boxscore quarter data of all games in a         particu, A function that return the player boxscore stats for all games in a         sing, A helper function that gets the boxscore data of a particular data.          Arg, A function that gets the boxscore quarterly data of a particular game. (+6 more)

### Community 4 - "Community 4"
Cohesion: 0.17
Nodes (12): A function that returns the game metadata, e.g. gamecodes of a round         in, A wrapper function for getting game data for all games in a single         round, A wrapper function with the all game data in a range of seasons          Args:, Concatenates the base URL and makes the game url.          Args:              se, A function that returns the game metadata, e.g. gamecodes of season          Arg, DataFrame, int, str (+4 more)

### Community 5 - "Community 5"
Cohesion: 0.23
Nodes (10): A function that gets the play-by-play data of *all* games in a range of, DataFrame, int, PlayByPlay, A function that gets the play-by-play data of *all* games in a range of, A class for getting the game play-by-play data.      Args:         competition (, Get the play-by-play (PBP) data enriched with the teams' lineups for         eve, A function that gets the play-by-play data of a particular game.          Args: (+2 more)

### Community 6 - "Community 6"
Cohesion: 0.29
Nodes (9): DataFrame, int, str, A wrapper function for collecting the leading players in a given         stat ca, A wrapper function for getting the players' stats for         - all seasons, The players' stats for *all* seasons.          Args:              endpoint (str), The players' stats for a *single* season.          Args:              endpoint (, The players' stats for a range of seasons.          Args:              endpoint (+1 more)

### Community 7 - "Community 7"
Cohesion: 0.40
Nodes (14): BoxScoreData Class, Shot Data Concept, A class for getting the player-level stats and data.      Args:         competit, Euroleague API README, EuroLeagueData Base Class, GameMetadata Class, GameStats Class, Euroleague API Module Index (+6 more)

### Community 8 - "Community 8"
Cohesion: 0.20
Nodes (7): calculate_fantasy_points(), calculate_form(), calculate_usage(), Feature Engineering for Fantasy Predictions Builds rolling averages, form indica, Calculate form indicators, Rolling usage trend from per-team-game minutes_rank (computed at df level), Calculate fantasy points based on official rules

### Community 9 - "Community 9"
Cohesion: 0.20
Nodes (9): colsample_bytree, gamma, learning_rate, max_depth, min_child_weight, n_estimators, reg_alpha, reg_lambda (+1 more)

### Community 10 - "Community 10"
Cohesion: 0.33
Nodes (3): looks_like_player_list(), Scrape player prices from EuroLeague Fantasy Challenge.  The site is a Flutter W, Walk a JSON blob and return a flat list of dicts that have     both a name field

### Community 11 - "Community 11"
Cohesion: 0.40
Nodes (6): Euroleague API Python Package, Euroleague Fantasy Competition ML Prediction, Pre-commit Configuration, Python Publish GitHub Workflow, Euroleague API Requirements, Euroleague Fantasy ML Project

### Community 12 - "Community 12"
Cohesion: 0.70
Nodes (4): main(), print_header(), Euroleague Fantasy Optimizer - Main Pipeline Run this to execute the full workfl, run_step()

## Knowledge Gaps
- **21 isolated node(s):** `Trial`, `str`, `Response`, `bool`, `int` (+16 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `EuroLeagueData` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 7`?**
  _High betweenness centrality (0.329) - this node is a cross-community bridge._
- **Why does `PlayerStats Class` connect `Community 7` to `Community 0`, `Community 3`, `Community 6`?**
  _High betweenness centrality (0.143) - this node is a cross-community bridge._
- **Why does `get_requests()` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 7`?**
  _High betweenness centrality (0.111) - this node is a cross-community bridge._
- **Are the 29 inferred relationships involving `EuroLeagueData` (e.g. with `BoxScoreData` and `int`) actually correct?**
  _`EuroLeagueData` has 29 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `BoxScoreData` (e.g. with `EuroLeagueData` and `PlayByPlay`) actually correct?**
  _`BoxScoreData` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `PlayerStats Class` (e.g. with `EuroLeagueData` and `Utils Module`) actually correct?**
  _`PlayerStats Class` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Feature Engineering for Fantasy Predictions Builds rolling averages, form indica`, `Calculate fantasy points based on official rules`, `Calculate form indicators` to the rest of the system?**
  _80 weakly-connected nodes found - possible documentation gaps or missing edges._