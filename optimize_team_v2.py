"""
Optimize fantasy team using ML predictions
"""

import pandas as pd
from pulp import *

print("=" * 60)
print("EUROLEAGUE FANTASY OPTIMIZER V2 (ML-POWERED)")
print("=" * 60)

# Load predictions
df = pd.read_csv('data/processed/next_round_predictions.csv')

# Filter to players with enough games (avoid flukes)
df = df[df['games_played_max'] >= 5].copy()

print(f"\nAvailable players: {len(df)}")
print(f"Using ML predictions from XGBoost model")

# Configuration
BUDGET = 100
ROSTER_SIZE = 10
MAX_PER_TEAM = 3

print(f"\nConstraints:")
print(f"  Budget: {BUDGET} credits")
print(f"  Roster: {ROSTER_SIZE} players")
print(f"  Max per team: {MAX_PER_TEAM}")

# Create optimization problem
prob = LpProblem("Fantasy_Team_ML", LpMaximize)

# Decision variables
player_vars = {}
for idx, row in df.iterrows():
    player_vars[idx] = LpVariable(f"player_{idx}", cat='Binary')

# Objective: Maximize PREDICTED fantasy points (not season average)
prob += lpSum([
    player_vars[idx] * df.loc[idx, 'predicted_fp']
    for idx in df.index
])

# Constraint 1: Roster size
prob += lpSum([player_vars[idx] for idx in df.index]) == ROSTER_SIZE

# Constraint 2: Budget
prob += lpSum([
    player_vars[idx] * df.loc[idx, 'price']
    for idx in df.index
]) <= BUDGET

# Constraint 3: Max 3 per team
for team in df['Team'].unique():
    team_players = df[df['Team'] == team].index
    prob += lpSum([player_vars[idx] for idx in team_players]) <= MAX_PER_TEAM

print("\nOptimizing...")
prob.solve(PULP_CBC_CMD(msg=0))

# Check if solution found
if prob.status != 1:
    print(f"\nWARNING: Optimization failed with status {prob.status}")
    print("This might happen if constraints are too tight")
else:
    print("✓ Optimal solution found")

# Extract results
selected_indices = [idx for idx in df.index if player_vars[idx].varValue == 1]
selected_team = df.loc[selected_indices].copy()
selected_team = selected_team.sort_values('predicted_fp', ascending=False)

print("\n" + "=" * 60)
print("OPTIMAL TEAM (ML Predictions)")
print("=" * 60)

print(f"\n{'Player':<25} {'Team':<5} {'Price':>6} {'L3':>6} {'Pred':>6} {'AvgSeason':>6}")
print("-" * 70)

total_cost = 0
total_predicted = 0
total_season_avg = 0

for _, p in selected_team.iterrows():
    print(f"{p['Player']:<25} {p['Team']:<5} {p['price']:>6.1f} "
          f"{p['fp_last_3']:>6.1f} {p['predicted_fp']:>6.1f} "
          f"{p['fantasy_points_mean']:>6.1f}")
    total_cost += p['price']
    total_predicted += p['predicted_fp']
    total_season_avg += p['fantasy_points_mean']

print("-" * 70)
print(f"{'TOTAL':<25} {'':<5} {total_cost:>6.1f} {'':<6} "
      f"{total_predicted:>6.1f} {total_season_avg:>6.1f}")

print(f"\nRemaining budget: {BUDGET - total_cost:.1f} credits")
print(f"Expected points (ML): {total_predicted:.1f}")
print(f"Expected points (season avg): {total_season_avg:.1f}")
print(f"ML uplift: {total_predicted - total_season_avg:+.1f} points")

# Save
selected_team[['Player', 'Team', 'price', 'fp_last_3', 'predicted_fp',
               'fantasy_points_mean', 'predicted_value']].to_csv(
    'optimal_team_ml.csv', index=False
)
print(f"\n✓ Team saved to: optimal_team_ml.csv")

# Compare with basic optimizer
print("\n" + "=" * 60)
print("COMPARISON: ML vs Season Average Picks")
print("=" * 60)

# Load old team if exists
try:
    old_team = pd.read_csv('optimal_team.csv')

    print("\nPlayers ADDED in ML version:")
    new_players = set(selected_team['Player']) - set(old_team['Player'])
    for player in new_players:
        p = selected_team[selected_team['Player'] == player].iloc[0]
        print(f"  + {p['Player']:<25} (Pred: {p['predicted_fp']:.1f}, "
              f"Last3: {p['fp_last_3']:.1f})")

    print("\nPlayers DROPPED from basic version:")
    dropped = set(old_team['Player']) - set(selected_team['Player'])
    for player in dropped:
        if player in df['Player'].values:
            p = df[df['Player'] == player].iloc[0]
            print(f"  - {p['Player']:<25} (Pred: {p['predicted_fp']:.1f}, "
                  f"Season: {p['fantasy_points_mean']:.1f})")
        else:
            print(f"  - {player}")

except FileNotFoundError:
    print("\nNo previous team found for comparison")

print("\n" + "=" * 60)
print("RISKY BUT HIGH UPSIDE ALTERNATIVES")
print("=" * 60)

# Find players with big prediction vs season average gap
df['upside'] = df['predicted_fp'] - df['fantasy_points_mean']
high_upside = df[~df.index.isin(selected_indices)].nlargest(5, 'upside')

print(f"\n{'Player':<25} {'Price':>6} {'Season':>7} {'Pred':>6} {'Upside':>7}")
print("-" * 60)
for _, p in high_upside.iterrows():
    print(f"{p['Player']:<25} {p['price']:>6.1f} "
          f"{p['fantasy_points_mean']:>7.1f} {p['predicted_fp']:>6.1f} "
          f"{p['upside']:>7.1f}")

print("\n" + "=" * 60)
print("DONE! Ready to submit your team")
print("=" * 60)