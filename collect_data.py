from pathlib import Path
from pybaseball import statcast

# Create a folder to store datasets
Path("data").mkdir(exist_ok=True)

# Download Statcast data for April 2025
print("Downloading April 2025 Statcast data...")

data = statcast(
    start_dt="2025-04-01",
    end_dt="2025-04-07"
)

# Display dataset information
print(f"Rows: {len(data)}")
print(f"Columns: {len(data.columns)}")
print(f"Unique batters: {data['batter'].nunique()}")

# Save the dataset locally
data.to_csv("data/statcast_april_2025_sample.csv", index=False)

print("Dataset saved successfully!")
