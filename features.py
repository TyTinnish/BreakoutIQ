from pathlib import Path
import pandas as pd

# Load previously downloaded Statcast data
data = pd.read_csv("data/statcast_april_2025_sample.csv")

# Keep balls put into play with recorded exit velocity
batted_balls = data[
    (data["type"] == "X") &
    (data["launch_speed"].notna())
].copy()

# Identify hard-hit balls (95+ mph)
batted_balls["hard_hit"] = (
    batted_balls["launch_speed"] >= 95
).astype(int)

# Calculate hitting statistics for each batter
player_stats = batted_balls.groupby("batter").agg(
    batted_balls=("launch_speed", "count"),
    avg_exit_velocity=("launch_speed", "mean"),
    hard_hit_rate=("hard_hit", "mean")
).reset_index()

# Convert hard-hit rate to percentage
player_stats["hard_hit_rate"] *= 100

# Round statistics for readability
player_stats = player_stats.round(2)

# Display the top 10 players by hard-hit rate
print("\nTop 10 Players by Hard-Hit Rate:")
print(
    player_stats.sort_values(
        "hard_hit_rate",
        ascending=False
    ).head(10).to_string(index=False)
)

# Save the processed dataset
Path("data").mkdir(exist_ok=True)
player_stats.to_csv(
    "data/player_features_april_2025_sample.csv",
    index=False
)

print("\nPlayer features saved successfully!")
