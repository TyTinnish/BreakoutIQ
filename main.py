from pybaseball import statcast_batter
import matplotlib.pyplot as plt
from pathlib import Path

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

# Filter for batted-ball events with recorded exit velocity
batted_balls = data[
    (data["type"] == "X") &
    (data["launch_speed"].notna())
].copy()

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

#--------------------------------------------------------#
#                GRAPHING EXIT VELOCITY                  #
#--------------------------------------------------------#

# Create a histogram of exit velocities
plt.figure(figsize=(10, 6))

plt.hist(
    batted_balls["launch_speed"],
    bins=20,
    edgecolor="black"
)

# Mark the hard-hit threshold
plt.axvline(
    x=95,
    color="red",
    linestyle="--",
    label="Hard-hit threshold (95 mph)"
)

plt.title("Aaron Judge - Exit Velocity Distribution (April 2025)")
plt.xlabel("Exit Velocity (mph)")
plt.ylabel("Number of Recorded Contacts")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

# Create the graphs folder if it doesn't exist
Path("ExitVelocityGraphs").mkdir(exist_ok=True)

# Save the graph inside the folder
plt.savefig("ExitVelocityGraphs/exit_velocity_distribution.png", dpi=300)

plt.show()
