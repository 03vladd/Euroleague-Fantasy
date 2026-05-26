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

# Load game-by-game data (combined Euroleague + Eurocup)
df = pd.read_csv('data/raw/all_player_game_stats_2025.csv')
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

# Apply official win/loss ±10% bonus
# Determine winning team per game by summing each team's points
team_points = df.groupby(['Gamecode', 'Team'])['Points'].sum().reset_index()
winners = (team_points
           .loc[team_points.groupby('Gamecode')['Points'].idxmax()]
           [['Gamecode', 'Team']]
           .rename(columns={'Team': 'winner'}))
df = df.merge(winners, on='Gamecode', how='left')
df['team_won'] = (df['Team'] == df['winner']).astype(int)
df['fantasy_points'] = df['fantasy_points'] * np.where(df['team_won'] == 1, 1.1, 0.9)
df = df.drop(columns=['winner'])

# Compute points_share at dataset level (share of team scoring in that game)
df['team_round_points'] = df.groupby(['Gamecode', 'Team'])['Points'].transform('sum')
df['points_share'] = (df['Points'] / df['team_round_points'].replace(0, np.nan)).fillna(0)

# Compute minutes_rank within each team-game (best-to-worst minutes)
df['minutes_rank'] = df.groupby(['Gamecode', 'Team'])['Minutes'].rank(ascending=False)

# Ensure numeric before rolling
df['IsStarter'] = pd.to_numeric(df['IsStarter'], errors='coerce').fillna(0)
df['Plusminus'] = pd.to_numeric(df['Plusminus'], errors='coerce').fillna(0)

# Sort by player and round
df = df.sort_values(['Player', 'Round'])

print(f"✓ Fantasy points calculated")

print("\n" + "=" * 60)
print("STEP 1: Rolling Averages (Recent Form)")
print("=" * 60)

def calculate_rolling_stats(group, windows=[3, 5, 10]):
    for window in windows:
        group[f'fp_last_{window}'] = group['fantasy_points'].rolling(window=window, min_periods=1).mean()
        group[f'fp_std_{window}'] = group['fantasy_points'].rolling(window=window, min_periods=2).std().fillna(0)
        group[f'minutes_last_{window}'] = group['Minutes'].rolling(window=window, min_periods=1).mean()

    # Starter rate: share of recent games the player started (predicts minutes)
    group['starter_rate_3'] = group['IsStarter'].rolling(3, min_periods=1).mean()
    group['starter_rate_5'] = group['IsStarter'].rolling(5, min_periods=1).mean()

    # Plus/minus rolling: net team score when player is on court
    group['plusminus_last_3'] = group['Plusminus'].rolling(3, min_periods=1).mean()
    group['plusminus_last_5'] = group['Plusminus'].rolling(5, min_periods=1).mean()

    # DNP risk: fraction of recent games with 0 minutes
    group['dnp'] = (group['Minutes'] == 0).astype(int)
    group['dnp_rate_last5']  = group['dnp'].rolling(5,  min_periods=1).mean()
    group['dnp_rate_last10'] = group['dnp'].rolling(10, min_periods=1).mean()

    # Home advantage: cumulative home-FP mean minus away-FP mean (no leakage)
    home_mean = group['fantasy_points'].where(group['Home'] == 1).expanding().mean()
    away_mean = group['fantasy_points'].where(group['Home'] == 0).expanding().mean()
    group['home_advantage'] = (home_mean - away_mean).fillna(0)

    return group

# Apply rolling stats per player
df = df.groupby('Player', group_keys=False).apply(calculate_rolling_stats)

print("✓ Calculated rolling averages (last 3, 5, 10 games) + starter rate + plus/minus")

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
print("STEP 3: Opponent Strength (rolling, no future leakage)")
print("=" * 60)

# Build opponent map: for each (Gamecode, Team) → Opponent
game_team_pairs = df.groupby('Gamecode')['Team'].unique().reset_index()
opponent_rows = []
for _, row in game_team_pairs.iterrows():
    teams = row['Team']
    if len(teams) == 2:
        opponent_rows.append({'Gamecode': row['Gamecode'], 'Team': teams[0], 'Opponent': teams[1]})
        opponent_rows.append({'Gamecode': row['Gamecode'], 'Team': teams[1], 'Opponent': teams[0]})
opponent_map = pd.DataFrame(opponent_rows)

# Avg FP per player per team per game, with Round attached
game_round = df[['Gamecode', 'Round']].drop_duplicates()
team_game_fp = (df.groupby(['Gamecode', 'Team'])['fantasy_points']
                .mean().reset_index()
                .merge(game_round, on='Gamecode', how='left')
                .merge(opponent_map, on=['Gamecode', 'Team'], how='left'))

# For each defending team (Opponent column), compute rolling 5-game defensive rating
# using only PRIOR rounds (shift(1) removes current game → no leakage)
rolling_parts = []
for defending_team, grp in team_game_fp.groupby('Opponent'):
    grp = grp.sort_values('Round').copy()
    grp['opp_defensive_rating'] = (grp['fantasy_points']
                                   .shift(1)
                                   .rolling(5, min_periods=1)
                                   .mean())
    rolling_parts.append(grp[['Gamecode', 'Opponent', 'opp_defensive_rating']])

rolling_def = pd.concat(rolling_parts, ignore_index=True)

# Attach opponent + rolling defensive rating to every player row
df = df.merge(opponent_map, on=['Gamecode', 'Team'], how='left')
df = df.merge(rolling_def, on=['Gamecode', 'Opponent'], how='left')

print("✓ Added rolling opponent defensive rating (5-game window, prior rounds only)")

print("\n" + "=" * 60)
print("STEP 4: Usage & Role Indicators")
print("=" * 60)

def calculate_usage(group):
    """Rolling usage trend from per-team-game minutes_rank (computed at df level)"""
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
    'fantasy_points',
    'fp_last_3', 'fp_last_5', 'fp_last_10',
    'fp_std_3', 'fp_std_5', 'fp_std_10',
    'minutes_last_3', 'minutes_last_5', 'minutes_last_10',
    'minutes_rank', 'usage_trend',
    'starter_rate_3', 'starter_rate_5',
    'plusminus_last_3', 'plusminus_last_5',
    'form_trend', 'hot_streak',
    'opp_defensive_rating',
    'team_won', 'Home',
    'points_share',
    'games_played',
    'dnp_rate_last5', 'dnp_rate_last10', 'home_advantage',
    'Minutes', 'Points', 'TotalRebounds', 'Assistances', 'Valuation'  # API returns 'Assistances' not 'Assists'
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