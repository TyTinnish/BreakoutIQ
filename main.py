from pybaseball import statcast_batter

#--------------------------------------------------------#
#                     AARON JUDGE DATA                   #
#--------------------------------------------------------#

# Aaron Judge's MLB player ID
PLAYER_ID = 592450

# Retrieve pitch-by-pitch data for April 2025
data = statcast_batter(
    "2025-04-01",
    "2025-04-30",
    player_id=PLAYER_ID
)

# Display basic information about the dataset
print("Number of rows:", len(data))
print("Number of columns:", len(data.columns))

print("\nFirst five rows:")
print(data.head())

print("\nAvailable columns:")
print(data.columns.tolist())

# Save a local copy for further analysis
data.to_csv("judge_statcast.csv", index=False)

#--------------------------------------------------------#
#               AARON JUDGE EXIT VELOCITY                #
#--------------------------------------------------------#

# Only keep pitches with recorded exit velocities
batted_balls = data.dropna(subset=["launch_speed"])

# Calculate average exit velocity
avg_exit_velocity = batted_balls["launch_speed"].mean()

# Calculate hard-hit rate (95+ mph)
hard_hit_balls = batted_balls[
    batted_balls["launch_speed"] >= 95
]

hard_hit_rate = (
    len(hard_hit_balls) / len(batted_balls)
) * 100

print("\n--- Hitting Statistics ---")
print(f"Average exit velocity: {avg_exit_velocity:.2f} mph")
print(f"Hard-hit rate: {hard_hit_rate:.2f}%")
print(f"Tracked batted balls: {len(batted_balls)}")
