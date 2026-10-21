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

#--------------------------------------------------------#
#                    STRIKEOUT RATE                      #
#--------------------------------------------------------#

# Keep the final pitch of each completed plate appearance
plate_appearances = data[
    data["events"].notna()
].copy()

# Identify strikeouts and walks
plate_appearances["strikeout"] = (
    plate_appearances["events"].isin([
        "strikeout",
        "strikeout_double_play"
    ])
).astype(int)

plate_appearances["walk"] = (
    plate_appearances["events"].isin([
        "walk",
        "intent_walk"
    ])
).astype(int)

# Calculate plate discipline statistics
discipline_stats = plate_appearances.groupby("batter").agg(
    plate_appearances=("events", "count"),
    strikeout_rate=("strikeout", "mean"),
    walk_rate=("walk", "mean")
).reset_index()

# Convert rates into percentages
discipline_stats["strikeout_rate"] *= 100
discipline_stats["walk_rate"] *= 100

# Combine contact statistics with plate discipline statistics
player_stats = player_stats.merge(
    discipline_stats,
    on="batter",
    how="left"
)

# Round the final statistics
player_stats = player_stats.round(2)

#--------------------------------------------------------#
#         HARD-HIT RATE AMONG QUALIFIED HITTERS          #
#--------------------------------------------------------#

# Only rank players with at least 10 batted balls
qualified_players = player_stats[
    player_stats["batted_balls"] >= 10
]

print("\nTop 10 Qualified Players by Hard-Hit Rate:")
print(
    qualified_players.sort_values(
        "hard_hit_rate",
        ascending=False
    ).head(10).to_string(index=False)
)

#--------------------------------------------------------#
#                       SAVE DATA                        #
#--------------------------------------------------------#

# Save the processed dataset
Path("data").mkdir(exist_ok=True)
player_stats.to_csv(
    "data/player_features_april_2025_sample.csv",
    index=False
)

print("\nPlayer features saved successfully!")
