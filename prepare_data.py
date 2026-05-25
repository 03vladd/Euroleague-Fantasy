"""
Prepare data including both Euroleague and Eurocup players
"""

import pandas as pd
import numpy as np
from pathlib import Path
from euroleague_api.boxscore_data import BoxScoreData
from euroleague_api.player_stats import PlayerStats

Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 1: Collecting Euroleague game-by-game stats...")
print("=" * 60)

# Check cache first
cache_file = Path('data/raw/all_player_game_stats_2025.csv')

if cache_file.exists():
    print(f"Found cached data, loading from {cache_file}")
    all_games = pd.read_csv(cache_file)
    print(f"✓ Loaded {len(all_games)} cached game performances")
else:
    print("No cache found, fetching from API...")
    boxscore_el = BoxScoreData("E")
    euroleague_games = boxscore_el.get_players_boxscore_stats_single_season(2025)
    print(f"✓ Euroleague: {len(euroleague_games)} game performances")

    print("\n" + "=" * 60)
    print("STEP 2: Collecting Eurocup game-by-game stats...")
    print("=" * 60)

    boxscore_ec = BoxScoreData("U")
    eurocup_games = boxscore_ec.get_players_boxscore_stats_single_season(2025)
    print(f"✓ Eurocup: {len(eurocup_games)} game performances")

    # Add competition marker
    euroleague_games['Competition'] = 'Euroleague'
    eurocup_games['Competition'] = 'Eurocup'

    # Combine
    all_games = pd.concat([euroleague_games, eurocup_games], ignore_index=True)
    print(f"\nCombined: {len(all_games)} total game performances")

    all_games.to_csv(cache_file, index=False)
    print(f"✓ Cached to {cache_file}")

print("\n" + "=" * 60)
print("STEP 3: Calculating fantasy points...")
print("=" * 60)

def calculate_fantasy_points(df):
    df = df.copy()

    fp = (
        df['Points'] +
        df['TotalRebounds'] +
        df['Assistances'] +
        df['Steals'] +
        df['BlocksFavour'] +
        df['FoulsReceived']
    )

    fp -= (
        df['Turnovers'] +
        df['BlocksAgainst'] +
        df['FoulsCommited']
    )

    missed_fg = (
        (df['FieldGoalsAttempted2'] - df['FieldGoalsMade2']) +
        (df['FieldGoalsAttempted3'] - df['FieldGoalsMade3'])
    )
    missed_ft = df['FreeThrowsAttempted'] - df['FreeThrowsMade']

    fp -= (missed_fg + missed_ft)
    df['fantasy_points'] = fp

    return df

def convert_minutes(time_str):
    if pd.isna(time_str):
        return 0
    try:
        parts = str(time_str).split(':')
        return int(parts[0]) + int(parts[1]) / 60
    except:
        return 0

# Filter team totals
all_games = all_games[all_games['Player'] != 'Total'].copy()
print(f"After removing team totals: {len(all_games)} performances")

all_games = calculate_fantasy_points(all_games)
all_games['Minutes'] = all_games['Minutes'].apply(convert_minutes)

print(f"✓ Calculated fantasy points for all games")

print("\n" + "=" * 60)
print("STEP 4: Aggregating season stats...")
print("=" * 60)

# Cache for season stats
season_cache = Path('data/raw/all_season_stats_2025.csv')

if season_cache.exists():
    print(f"Found cached season stats, loading from {season_cache}")
    all_season_stats = pd.read_csv(season_cache)
    print(f"✓ Loaded {len(all_season_stats)} cached season stats")
else:
    print("No cache found, fetching season stats from API...")
    # Get season totals from PlayerStats API (has aggregated stats)
    player_stats_el = PlayerStats("E")
    player_stats_ec = PlayerStats("U")

    el_season = player_stats_el.get_player_stats_single_season("traditional", 2025)
    ec_season = player_stats_ec.get_player_stats_single_season("traditional", 2025)

    el_season['Competition'] = 'Euroleague'
    ec_season['Competition'] = 'Eurocup'

    # Combine season stats
    all_season_stats = pd.concat([el_season, ec_season], ignore_index=True)

    all_season_stats.to_csv(season_cache, index=False)
    print(f"✓ Cached to {season_cache}")

# Keep important columns
important_cols = [
    'playerRanking', 'gamesPlayed', 'gamesStarted', 'minutesPlayed',
    'pointsScored', 'twoPointersMade', 'twoPointersAttempted',
    'twoPointersPercentage', 'threePointersMade', 'threePointersAttempted',
    'threePointersPercentage', 'freeThrowsMade', 'freeThrowsAttempted',
    'freeThrowsPercentage', 'offensiveRebounds', 'defensiveRebounds',
    'totalRebounds', 'assists', 'steals', 'turnovers', 'blocks',
    'blocksAgainst', 'foulsCommited', 'foulsDrawn', 'pir',
    'player.code', 'player.name', 'team.name'
]

available_cols = [col for col in important_cols if col in all_season_stats.columns]
stats_clean = all_season_stats[available_cols].copy()

print(f"✓ Season stats: {len(stats_clean)} players")

print("\n" + "=" * 60)
print("STEP 5: Merging with prices...")
print("=" * 60)

prices = pd.read_csv('euroleague_prices.csv')
print(f"Prices: {len(prices)} players")

# Merge with prices
stats_with_prices = pd.merge(
    stats_clean,
    prices,
    on='player.name',
    how='right'
)

print(f"Merged: {len(stats_with_prices)} players")
print(f"  With stats: {stats_with_prices['gamesPlayed'].notna().sum()}")
print(f"  Missing stats: {stats_with_prices['gamesPlayed'].isna().sum()}")

# Calculate fantasy points from season stats
# Note: PlayerStats API uses different column names than BoxScore
def calculate_fp_from_season_stats(df):
    df = df.copy()

    fp = (
        df['pointsScored'] +
        df['totalRebounds'] +
        df['assists'] +
        df['steals'] +
        df['blocks'] +
        df['foulsDrawn']
    )

    fp -= (
        df['turnovers'] +
        df['blocksAgainst'] +
        df['foulsCommited']
    )

    # Missed shots
    missed_fg = (
        (df['twoPointersAttempted'] - df['twoPointersMade']) +
        (df['threePointersAttempted'] - df['threePointersMade'])
    )
    missed_ft = df['freeThrowsAttempted'] - df['freeThrowsMade']

    fp -= (missed_fg + missed_ft)

    df['fantasy_points_total'] = fp
    df['fantasy_points_per_game'] = fp / df['gamesPlayed']

    return df

stats_with_prices = calculate_fp_from_season_stats(stats_with_prices)

# Calculate value
stats_with_prices['value_score'] = (
    stats_with_prices['fantasy_points_per_game'] / stats_with_prices['price']
)

stats_with_prices.to_csv('data/processed/all_players_with_prices.csv', index=False)

print("\n" + "=" * 60)
print("COVERAGE ANALYSIS")
print("=" * 60)

# Count by competition
el_count = len(all_season_stats[all_season_stats['Competition'] == 'Euroleague'])
ec_count = len(all_season_stats[all_season_stats['Competition'] == 'Eurocup'])

print(f"\nPlayers in price list: {len(prices)}")
print(f"Players with Euroleague stats: {el_count}")
print(f"Players with Eurocup stats: {ec_count}")
print(f"Players with either: {len(all_season_stats)}")
print(f"Players with prices but no stats: {stats_with_prices['gamesPlayed'].isna().sum()}")

# Show players with stats but no price
players_with_stats = set(all_season_stats['player.name'])
players_with_prices = set(prices['player.name'])

no_price = players_with_stats - players_with_prices
print(f"\nPlayers with stats but no price: {len(no_price)}")
if len(no_price) > 0 and len(no_price) < 20:
    print("Examples:", list(no_price)[:10])

print("\n" + "=" * 60)
print("TOP 10 BY VALUE (All competitions)")
print("=" * 60)

valid_stats = stats_with_prices[stats_with_prices['gamesPlayed'].notna()].copy()
top_value = valid_stats.nlargest(10, 'value_score')

print(f"\n{'Player':<25} {'Team':<10} {'Games':>6} {'FP/G':>6} {'Price':>6} {'Value':>6}")
print("-" * 75)
for _, p in top_value.iterrows():
    team = p.get('team.name', p.get('Team', 'N/A'))
    print(f"{p['player.name']:<25} {str(team):<10} "
          f"{int(p['gamesPlayed']):>6} {p['fantasy_points_per_game']:>6.1f} "
          f"{p['price']:>6.1f} {p['value_score']:>6.3f}")

print("\n" + "=" * 60)
print("DONE! Next: Run build_features.py with combined data")
print("=" * 60)