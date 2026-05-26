"""
Feature Engineering for Fantasy Predictions
Builds rolling averages, form indicators, opponent strength, etc.
"""

import pandas as pd
import numpy as np
from pathlib import Path

try:
    from euroleague_api.schedule import Schedule as _ELSchedule
    _SCHEDULE_AVAILABLE = True
except ImportError:
    _SCHEDULE_AVAILABLE = False

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

# Merge player positions (G/F/C) from the fantasy analysis file
_POSITION_MAP = {'Guard': 'G', 'Forward': 'F', 'Center': 'C'}
try:
    _pos = (pd.read_csv('euroleague_fantasy_analysis_complete.csv')
            [['player_name', 'estimated_position']]
            .drop_duplicates('player_name')
            .rename(columns={'player_name': 'Player', 'estimated_position': 'Position'}))
    _pos['Position'] = _pos['Position'].map(_POSITION_MAP).fillna('?')
    df = df.merge(_pos, on='Player', how='left')
    df['Position'] = df['Position'].fillna('?')
    print(f"Position data: {(_pos['Position'] != '?').sum()} players mapped")
except FileNotFoundError:
    df['Position'] = '?'
    print("No position file — position-specific defense disabled")

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

    # Consistency (coefficient of variation): lower = more reliable for captain pick
    group['fp_cv_5']  = group['fp_std_5']  / (group['fp_last_5'].abs()  + 1e-6)
    group['fp_cv_10'] = group['fp_std_10'] / (group['fp_last_10'].abs() + 1e-6)

    return group

# Apply rolling stats per player
df = df.groupby('Player', group_keys=False).apply(calculate_rolling_stats)

print("✓ Calculated rolling averages (last 3, 5, 10 games) + starter rate + plus/minus")

print("\n" + "=" * 60)
print("STEP 1b: Rest / Fatigue Features (days_rest, short_rest)")
print("=" * 60)

if _SCHEDULE_AVAILABLE:
    try:
        _sched = _ELSchedule('E').get_schedule(2025)
        _sched['game_date'] = pd.to_datetime(_sched['date'], format='%b %d, %Y')
        _home = _sched[['gameday', 'homecode', 'game_date']].rename(
            columns={'gameday': 'Round', 'homecode': 'Team'})
        _away = _sched[['gameday', 'awaycode', 'game_date']].rename(
            columns={'gameday': 'Round', 'awaycode': 'Team'})
        _team_dates = (pd.concat([_home, _away])
                       .drop_duplicates(['Round', 'Team'])
                       .sort_values(['Team', 'game_date'])   # sort by actual date, not round
                       .reset_index(drop=True))
        # clip(lower=0) handles rescheduled games played out of round order
        _team_dates['days_rest'] = (_team_dates.groupby('Team')['game_date']
                                    .diff().dt.days.fillna(7).clip(lower=0))
        # short_rest = inter-round gap ≤ 3 days (compressed EL scheduling, not NBA B2Bs)
        _team_dates['short_rest'] = (_team_dates['days_rest'] <= 3).astype(int)
        df = df.merge(_team_dates[['Round', 'Team', 'days_rest', 'short_rest']],
                      on=['Round', 'Team'], how='left')
        df['days_rest']    = df['days_rest'].fillna(7)
        df['short_rest'] = df['short_rest'].fillna(0)
        print(f"✓ Added days_rest + short_rest  "
              f"(B2B games: {df['short_rest'].sum():.0f})")
    except Exception as e:
        df['days_rest']    = 7
        df['short_rest'] = 0
        print(f"  Schedule fetch failed ({e}) — days_rest=7, short_rest=0")
else:
    df['days_rest']    = 7
    df['short_rest'] = 0
    print("  Schedule API unavailable — days_rest=7, short_rest=0")

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

# Position-specific defensive rating: avg FP allowed to G/F/C per opponent team
if df['Position'].ne('?').any():
    pos_game_fp = (df[df['Position'] != '?']
                   .groupby(['Gamecode', 'Round', 'Opponent', 'Position'])['fantasy_points']
                   .mean().reset_index()
                   .rename(columns={'Opponent': 'def_team', 'fantasy_points': 'fp_vs_pos'}))
    pos_parts = []
    for (def_team, pos), grp in pos_game_fp.groupby(['def_team', 'Position']):
        grp = grp.sort_values('Round').copy()
        grp['opp_def_pos_rating'] = grp['fp_vs_pos'].shift(1).rolling(5, min_periods=1).mean()
        pos_parts.append(grp[['Gamecode', 'def_team', 'Position', 'opp_def_pos_rating']])
    pos_def = pd.concat(pos_parts, ignore_index=True).rename(columns={'def_team': 'Opponent'})
    df = df.merge(pos_def, on=['Gamecode', 'Opponent', 'Position'], how='left')
    # Fall back to team-level rating for unknown positions
    df['opp_def_pos_rating'] = df['opp_def_pos_rating'].fillna(df['opp_defensive_rating'])
    print("✓ Added position-specific opponent defensive rating (G/F/C split)")
else:
    df['opp_def_pos_rating'] = df['opp_defensive_rating']
    print("  No position data — opp_def_pos_rating = opp_defensive_rating")

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
    'fp_cv_5', 'fp_cv_10',
    'days_rest', 'short_rest',
    'opp_def_pos_rating',
    'Minutes', 'Points', 'TotalRebounds', 'Assistances', 'Valuation'
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

# Load prices — prefer processed file (full season coverage) over scraper output
_price_path = ('data/processed/all_players_with_prices.csv'
               if Path('data/processed/all_players_with_prices.csv').exists()
               else 'euroleague_prices.csv')
prices = pd.read_csv(_price_path)[['player.name', 'price']].drop_duplicates('player.name')
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