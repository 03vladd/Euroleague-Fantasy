"""
Prepare Euroleague fantasy data
Run this first to get your dataset ready
"""

import pandas as pd
import numpy as np
from euroleague_api.boxscore_data import BoxScoreData
from euroleague_api.game_stats import GameStats
from pathlib import Path

# Create directories
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 1: Collecting game-by-game player stats...")
print("=" * 60)

boxscore = BoxScoreData("E")
game_by_game = boxscore.get_player_boxscore_stats_single_season(2024)

print(f"✓ Collected {len(game_by_game)} game performances")
print(f"  Columns: {list(game_by_game.columns)}")

game_by_game.to_csv('data/raw/player_game_stats_2024.csv', index=False)

print("\n" + "=" * 60)
print("STEP 2: Calculating fantasy points...")
print("=" * 60)

def calculate_fantasy_points(df):
    """Calculate fantasy points based on official rules"""
    df = df.copy()

    # Positive stats (+1 each)
    fp = (
        df['Points'] +
        df['TotalRebounds'] +
        df['Assistances'] +
        df['Steals'] +
        df['BlocksFavour'] +
        df['FoulsReceived']
    )

    # Negative stats (-1 each)
    fp -= (
        df['Turnovers'] +
        df['BlocksAgainst'] +
        df['FoulsCommited']
    )

    # Missed shots
    missed_fg = (
        (df['FieldGoalsAttempted2'] - df['FieldGoalsMade2']) +
        (df['FieldGoalsAttempted3'] - df['FieldGoalsMade3'])
    )
    missed_ft = df['FreeThrowsAttempted'] - df['FreeThrowsMade']

    fp -= (missed_fg + missed_ft)

    df['fantasy_points'] = fp

    return df

game_by_game = calculate_fantasy_points(game_by_game)

# Convert Minutes from MM:SS string to decimal
def convert_minutes(time_str):
    """Convert MM:SS to decimal minutes"""
    if pd.isna(time_str):
        return 0
    try:
        parts = str(time_str).split(':')
        return int(parts[0]) + int(parts[1]) / 60
    except:
        return 0

game_by_game['Minutes'] = game_by_game['Minutes'].apply(convert_minutes)

print(f"✓ Calculated fantasy points for all games")
print("\nTop 5 single-game performances:")
top5 = game_by_game.nlargest(5, 'fantasy_points')[
    ['Player', 'Team', 'Round', 'fantasy_points', 'Points', 'TotalRebounds']
]
print(top5.to_string(index=False))

print("\n" + "=" * 60)
print("STEP 3: Aggregating to season averages...")
print("=" * 60)

player_season_stats = game_by_game.groupby('Player').agg({
    'fantasy_points': ['mean', 'std', 'sum', 'max'],
    'Minutes': 'mean',
    'Points': 'mean',
    'TotalRebounds': 'mean',
    'Assistances': 'mean',
    'Valuation': 'mean',
    'Team': 'first',
    'Round': 'count'
}).reset_index()

# Flatten column names
player_season_stats.columns = ['_'.join(col).strip('_') for col in player_season_stats.columns]
player_season_stats.rename(columns={'Round_count': 'games_played'}, inplace=True)

print(f"✓ Aggregated stats for {len(player_season_stats)} players")

print("\n" + "=" * 60)
print("STEP 4: Merging with prices...")
print("=" * 60)

prices = pd.read_csv('euroleague_prices.csv')
print(f"✓ Loaded prices for {len(prices)} players")

final_df = pd.merge(
    player_season_stats,
    prices,
    left_on='Player',
    right_on='player.name',
    how='right'
)

print(f"✓ Merged dataset: {len(final_df)} total players")
print(f"  Players with stats: {final_df['fantasy_points_mean'].notna().sum()}")
print(f"  Players missing stats: {final_df['fantasy_points_mean'].isna().sum()}")

print("\n" + "=" * 60)
print("STEP 5: Calculating value metrics...")
print("=" * 60)

# Filter to players with stats
final_df = final_df[final_df['fantasy_points_mean'].notna()].copy()

# Calculate value
final_df['value_score'] = final_df['fantasy_points_mean'] / final_df['price']
final_df['consistency'] = 1 / (final_df['fantasy_points_std'] + 1)
final_df['points_per_credit'] = final_df['fantasy_points_mean'] / final_df['price']

print(f"✓ Final dataset ready: {len(final_df)} players with complete data")

print("\n" + "=" * 60)
print("STEP 6: Saving results...")
print("=" * 60)

final_df.to_csv('data/processed/fantasy_dataset_2024.csv', index=False)
print("✓ Saved to: data/processed/fantasy_dataset_2024.csv")

print("\n" + "=" * 60)
print("TOP 20 PLAYERS BY VALUE SCORE")
print("=" * 60)

top_value = final_df.nlargest(20, 'value_score')[[
    'Player', 'Team_first', 'games_played', 'fantasy_points_mean',
    'price', 'value_score'
]]
print(top_value.to_string(index=False))

print("\n" + "=" * 60)
print("TOP 20 PLAYERS BY TOTAL FANTASY POINTS")
print("=" * 60)

top_total = final_df.nlargest(20, 'fantasy_points_mean')[[
    'Player', 'Team_first', 'games_played', 'fantasy_points_mean',
    'price', 'value_score'
]]
print(top_total.to_string(index=False))

print("\n" + "=" * 60)
print("DONE! Next step: Run the optimizer")
print("=" * 60)