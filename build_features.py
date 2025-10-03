"""
Feature Engineering for Fantasy Predictions
Builds rolling averages, form indicators, opponent strength, etc.
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Load game-by-game data
df = pd.read_csv('data/raw/player_game_stats_2024.csv')
print(f"\nLoaded {len(df)} game performances")

# Filter out team totals
df = df[df['Player'] != 'Total'].copy()
print(f"After removing team totals: {len(df)} performances")

# Calculate fantasy points first
print("\nCalculating fantasy points...")

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

df = calculate_fantasy_points(df)

# Convert Minutes from MM:SS to decimal
def convert_minutes(time_str):
    if pd.isna(time_str):
        return 0
    try:
        parts = str(time_str).split(':')
        return int(parts[0]) + int(parts[1]) / 60
    except:
        return 0

df['Minutes'] = df['Minutes'].apply(convert_minutes)

# Sort by player and round
df = df.sort_values(['Player', 'Round'])

print(f"✓ Fantasy points calculated")

print("\n" + "=" * 60)
print("STEP 1: Rolling Averages (Recent Form)")
print("=" * 60)

def calculate_rolling_stats(group, windows=[3, 5, 10]):
    """Calculate rolling averages for a player"""
    for window in windows:
        # Rolling mean
        group[f'fp_last_{window}'] = group['fantasy_points'].rolling(
            window=window, min_periods=1
        ).mean()

        # Rolling std (consistency)
        group[f'fp_std_{window}'] = group['fantasy_points'].rolling(
            window=window, min_periods=2
        ).std().fillna(0)

        # Minutes trend
        group[f'minutes_last_{window}'] = group['Minutes'].rolling(
            window=window, min_periods=1
        ).mean()

    return group

# Apply rolling stats per player
df = df.groupby('Player', group_keys=False).apply(calculate_rolling_stats)

print("✓ Calculated rolling averages (last 3, 5, 10 games)")

print("\n" + "=" * 60)
print("STEP 2: Form Indicators")
print("=" * 60)

def calculate_form(group):
    """Calculate form indicators"""
    # Games played
    group['games_played'] = range(1, len(group) + 1)

    # Trend: compare recent vs earlier performance
    group['form_trend'] = group['fp_last_3'] - group['fantasy_points'].expanding().mean()

    # Hot streak: 3+ games above season average
    season_avg = group['fantasy_points'].mean()
    group['above_avg'] = (group['fantasy_points'] > season_avg).astype(int)
    group['hot_streak'] = group['above_avg'].rolling(3, min_periods=1).sum()

    return group

df = df.groupby('Player', group_keys=False).apply(calculate_form)

print("✓ Calculated form indicators")

print("\n" + "=" * 60)
print("STEP 3: Opponent Strength")
print("=" * 60)

# Calculate team defensive ratings (points allowed per game)
team_defense = df.groupby('Team').agg({
    'fantasy_points': 'mean'  # Average fantasy points scored by players on this team
}).rename(columns={'fantasy_points': 'team_offensive_rating'})

# For each game, get opponent's defensive strength
# This requires matching home/away - simplified version:
df = df.merge(
    team_defense,
    left_on='Team',
    right_index=True,
    how='left'
)

print("✓ Added team strength metrics")

print("\n" + "=" * 60)
print("STEP 4: Usage & Role Indicators")
print("=" * 60)

def calculate_usage(group):
    """Calculate player's role/usage on team"""
    # Minutes share (approximation - would need team totals for exact)
    group['minutes_rank'] = group.groupby('Round')['Minutes'].rank(ascending=False)

    # Scoring role
    group['points_share'] = group['Points'] / group.groupby('Round')['Points'].transform('sum')

    # Usage trend
    group['usage_trend'] = group['minutes_rank'].rolling(5, min_periods=1).mean()

    return group

df = df.groupby('Player', group_keys=False).apply(calculate_usage)

print("✓ Calculated usage metrics")

print("\n" + "=" * 60)
print("STEP 5: Save Enhanced Dataset")
print("=" * 60)

# Save game-by-game with features
df.to_csv('data/processed/player_games_with_features.csv', index=False)
print("✓ Saved: data/processed/player_games_with_features.csv")

print("\n" + "=" * 60)
print("STEP 6: Aggregate Latest Features for Prediction")
print("=" * 60)

# Get most recent stats for each player (what we'd use for next round prediction)
latest_by_player = df.sort_values('Round').groupby('Player').tail(1)

# Aggregate key features
prediction_features = latest_by_player[[
    'Player', 'Team', 'Season', 'Phase',
    'fantasy_points',  # last game
    'fp_last_3', 'fp_last_5', 'fp_last_10',  # rolling averages
    'fp_std_3', 'fp_std_5',  # consistency
    'minutes_last_3', 'minutes_last_5',  # minutes trend
    'form_trend', 'hot_streak',  # form
    'team_offensive_rating',  # team strength
    'games_played',  # experience this season
    'Minutes', 'Points', 'Valuation'  # last game stats
]].copy()

# Overall season stats
season_stats = df.groupby('Player').agg({
    'fantasy_points': ['mean', 'std', 'max'],
    'Minutes': 'mean',
    'games_played': 'max'
}).reset_index()

season_stats.columns = ['_'.join(col).strip('_') for col in season_stats.columns]

# Merge
prediction_features = prediction_features.merge(
    season_stats,
    on='Player',
    how='left',
    suffixes=('', '_season')
)

# Load prices
prices = pd.read_csv('euroleague_prices.csv')
prediction_features = prediction_features.merge(
    prices,
    left_on='Player',
    right_on='player.name',
    how='inner'
)

print(f"✓ Created prediction dataset: {len(prediction_features)} players")

# Save
prediction_features.to_csv('data/processed/prediction_features.csv', index=False)
print("✓ Saved: data/processed/prediction_features.csv")

print("\n" + "=" * 60)
print("TOP 10 BY RECENT FORM (Last 3 Games)")
print("=" * 60)

top_form = prediction_features.nlargest(10, 'fp_last_3')[[
    'Player', 'Team', 'fp_last_3', 'form_trend', 'hot_streak', 'price'
]]
print(top_form.to_string(index=False))

print("\n" + "=" * 60)
print("TOP 10 MOST CONSISTENT (Low Variance)")
print("=" * 60)

# Lower std = more consistent
consistent = prediction_features.nsmallest(10, 'fp_std_5')[[
    'Player', 'Team', 'fp_last_5', 'fp_std_5', 'price'
]]
print(consistent.to_string(index=False))

print("\n" + "=" * 60)
print("DONE! Ready for modeling")
print("=" * 60)