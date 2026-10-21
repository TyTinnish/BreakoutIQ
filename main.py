from pybaseball import statcast_batter

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
