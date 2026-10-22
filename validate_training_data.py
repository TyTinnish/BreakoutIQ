
import pandas as pd

# Load the generated training dataset
data = pd.read_csv("data/training_2025.csv")

print("--- DATASET VALIDATION ---")

# 1. Check dimensions
print(f"Training examples: {len(data):,}")
print(f"Unique hitters: {data['batter'].nunique()}")

# 2. Check missing values
print("\nMissing values:")
print(data.isna().sum())

# 3. Check for duplicate training examples
duplicates = data.duplicated(
    subset=["batter", "season", "window_start"]
).sum()

print(f"\nDuplicate examples: {duplicates}")

# 4. Verify breakout labels
expected_labels = (
    data["future_ops"] - data["recent_ops"] >= 0.075
).astype(int)

label_errors = (
    expected_labels != data["breakout"]
).sum()

print(f"Incorrect breakout labels: {label_errors}")

# 5. Check basic statistical ranges
rate_columns = [
    "hard_hit_rate",
    "strikeout_rate",
    "walk_rate"
]

for column in rate_columns:
    invalid = (~data[column].between(0, 1)).sum()
    print(f"Invalid {column} values: {invalid}")

# 6. View summary statistics
print("\nFeature summary:")
print(
    data[
        ["avg_exit_velocity", "hard_hit_rate",
         "strikeout_rate", "walk_rate", "recent_ops"]
    ].describe().round(3)
)
