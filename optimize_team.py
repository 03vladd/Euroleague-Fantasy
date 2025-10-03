"""
Optimize fantasy team selection using linear programming
"""

import pandas as pd
from pulp import *

# Load data
df = pd.read_csv('data/processed/fantasy_dataset_2024.csv')

print("=" * 60)
print("EUROLEAGUE FANTASY TEAM OPTIMIZER")
print("=" * 60)

# Configuration
BUDGET = 100
ROSTER_SIZE = 10
GUARDS = 4
FORWARDS = 4
CENTERS = 2

print(f"\nConstraints:")
print(f"  Budget: {BUDGET} credits")
print(f"  Roster: {ROSTER_SIZE} players ({GUARDS}G, {FORWARDS}F, {CENTERS}C)")
print(f"  Available players: {len(df)}")

# Add position (you'll need to manually assign these or get from another source)
# For now, let's assume everyone can play any position
# TODO: Add actual positions from fantasy website

# Create optimization problem
prob = LpProblem("Fantasy_Team", LpMaximize)

# Decision variables: 1 if player is selected, 0 otherwise
player_vars = {}
for idx, row in df.iterrows():
    player_vars[idx] = LpVariable(f"player_{idx}", cat='Binary')

# Objective: Maximize expected fantasy points
prob += lpSum([
    player_vars[idx] * df.loc[idx, 'fantasy_points_mean']
    for idx in df.index
])

# Constraint 1: Total roster size
prob += lpSum([player_vars[idx] for idx in df.index]) == ROSTER_SIZE

# Constraint 2: Budget
prob += lpSum([
    player_vars[idx] * df.loc[idx, 'price']
    for idx in df.index
]) <= BUDGET

# Constraint 3: Max 3 players per team (if team data available)
if 'Team_first' in df.columns:
    for team in df['Team_first'].unique():
        team_players = df[df['Team_first'] == team].index
        prob += lpSum([player_vars[idx] for idx in team_players]) <= 3

print("\nOptimizing...")
prob.solve(PULP_CBC_CMD(msg=0))

# Extract selected players
selected_indices = [idx for idx in df.index if player_vars[idx].varValue == 1]
selected_team = df.loc[selected_indices].copy()

print("\n" + "=" * 60)
print("OPTIMAL TEAM")
print("=" * 60)

# Sort by fantasy points
selected_team = selected_team.sort_values('fantasy_points_mean', ascending=False)

print(f"\n{'Player':<25} {'Team':<5} {'Price':>6} {'FP/G':>6} {'Games':>6}")
print("-" * 60)

total_cost = 0
total_points = 0

for _, player in selected_team.iterrows():
    print(f"{player['Player']:<25} {player['Team_first']:<5} "
          f"{player['price']:>6.1f} {player['fantasy_points_mean']:>6.1f} "
          f"{int(player['games_played']):>6}")
    total_cost += player['price']
    total_points += player['fantasy_points_mean']

print("-" * 60)
print(f"{'TOTAL':<25} {'':<5} {total_cost:>6.1f} {total_points:>6.1f}")
print(f"\nRemaining budget: {BUDGET - total_cost:.1f} credits")
print(f"Expected total points: {total_points:.1f}")

# Save team
selected_team[['Player', 'Team_first', 'price', 'fantasy_points_mean',
               'games_played', 'value_score']].to_csv(
    'optimal_team.csv', index=False
)
print(f"\nTeam saved to: optimal_team.csv")

# Show some alternatives (next best players not selected)
print("\n" + "=" * 60)
print("TOP ALTERNATIVES (not selected)")
print("=" * 60)

alternatives = df[~df.index.isin(selected_indices)].nlargest(10, 'value_score')
print(f"\n{'Player':<25} {'Team':<5} {'Price':>6} {'FP/G':>6} {'Value':>6}")
print("-" * 60)
for _, player in alternatives.iterrows():
    print(f"{player['Player']:<25} {player['Team_first']:<5} "
          f"{player['price']:>6.1f} {player['fantasy_points_mean']:>6.1f} "
          f"{player['value_score']:>6.3f}")