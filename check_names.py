import pandas as pd

prices = pd.read_csv('euroleague_prices.csv')
stats = pd.read_csv('data/raw/all_season_stats_2024.csv')

# Top 20 most expensive players
top_expensive = prices.nlargest(20, 'price')

print("Top 20 expensive players - are they in stats?")
for _, p in top_expensive.iterrows():
    in_stats = p['player.name'] in stats['player.name'].values
    status = "✓" if in_stats else "✗"
    print(f"{status} {p['player.name']:<25} ${p['price']}")