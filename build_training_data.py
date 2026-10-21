from pathlib import Path
import pandas as pd
from pybaseball import statcast_batter

# Player and season to analyze
PLAYER_ID = 592450
SEASON = 2025

# Create the data directory
Path("data").mkdir(exist_ok=True)

# Download Aaron Judge's 2025 regular-season data
print("Downloading 2025 Statcast data...")

data = statcast_batter(
    "2025-03-27",
    "2025-09-28",
    player_id=PLAYER_ID
)

# Keep regular-season games
data = data[data["game_type"] == "R"].copy()

# Sort pitches by game and plate appearance
data["game_date"] = pd.to_datetime(data["game_date"])

data = data.sort_values([
    "game_date",
    "game_pk",
    "at_bat_number",
    "pitch_number"
])

print(f"Total pitches: {len(data)}")
print(f"Games played: {data['game_pk'].nunique()}")

#--------------------------------------------------------#
#                    CALCULATING OPS                     #
#--------------------------------------------------------#

# Events that count as hits
HIT_BASES = {
    "single": 1,
    "double": 2,
    "triple": 3,
    "home_run": 4
}

# Events excluded from official at-bats
NON_AB_EVENTS = {
    "walk",
    "intent_walk",
    "hit_by_pitch",
    "sac_fly",
    "sac_fly_double_play",
    "sac_bunt",
    "sac_bunt_double_play",
    "catcher_interf"
}

def calculate_ops(pitches):
    # One recorded outcome per completed plate appearance
    pa = pitches[pitches["events"].notna()].copy()

    events = pa["events"]

    # Count hits and total bases
    hits = events.isin(HIT_BASES).sum()
    total_bases = events.map(HIT_BASES).fillna(0).sum()

    # Walks, hit-by-pitches, and sacrifice flies
    walks = events.isin(["walk", "intent_walk"]).sum()
    hbp = (events == "hit_by_pitch").sum()
    sac_flies = events.isin([
        "sac_fly",
        "sac_fly_double_play"
    ]).sum()

    # Official at-bats
    at_bats = (~events.isin(NON_AB_EVENTS)).sum()

    # Avoid dividing by zero
    if at_bats == 0:
        return None

    obp_denominator = at_bats + walks + hbp + sac_flies

    if obp_denominator == 0:
        return None

    obp = (hits + walks + hbp) / obp_denominator
    slg = total_bases / at_bats

    return obp + slg

#--------------------------------------------------------#
#                COMPARING 30-GAME WINDOWS               #
#--------------------------------------------------------#

def calculate_features(pitches):
    # Completed plate appearances
    pa = pitches[pitches["events"].notna()]

    # Balls put into play with recorded exit velocity
    batted = pitches[
        (pitches["type"] == "X") &
        (pitches["launch_speed"].notna())
    ]

    if len(pa) == 0 or len(batted) == 0:
        return None

    # Calculate hitting metrics
    avg_ev = batted["launch_speed"].mean()
    hard_hit_rate = (batted["launch_speed"] >= 95).mean()

    strikeout_rate = pa["events"].isin([
        "strikeout",
        "strikeout_double_play"
    ]).mean()

    walk_rate = pa["events"].isin([
        "walk",
        "intent_walk"
    ]).mean()

    return {
        "avg_exit_velocity": avg_ev,
        "hard_hit_rate": hard_hit_rate,
        "strikeout_rate": strikeout_rate,
        "walk_rate": walk_rate,
        "plate_appearances": len(pa),
        "batted_balls": len(batted),
        "ops": calculate_ops(pitches)
    }


# Identify games in chronological order
games = (
    data[["game_date", "game_pk"]]
    .drop_duplicates()
    .sort_values(["game_date", "game_pk"])
    ["game_pk"]
    .tolist()
)

training_rows = []

# Slide forward 10 games between training examples
WINDOW_SIZE = 30
STEP_SIZE = 10

for start in range(
        0,
        len(games) - (2 * WINDOW_SIZE) + 1,
        STEP_SIZE
    ):
    previous_games = games[start:start + 30]
    future_games = games[start + 30:start + 60]

    # Get pitches from each window
    previous_data = data[
        data["game_pk"].isin(previous_games)
    ]

    future_data = data[
        data["game_pk"].isin(future_games)
    ]

    # Calculate past and future statistics
    previous = calculate_features(previous_data)
    future = calculate_features(future_data)

    if previous is None or future is None:
        continue

    # Require sufficient plate appearances
    if (
        previous["plate_appearances"] < 75 or
        future["plate_appearances"] < 75 or
        previous["ops"] is None or
        future["ops"] is None
    ):
        continue

    ops_change = future["ops"] - previous["ops"]

    # Define a breakout as improving OPS by 0.075+
    breakout = int(ops_change >= 0.075)

    training_rows.append({
        "batter": PLAYER_ID,
        "season": SEASON,
        "window_start": start + 1,
        "avg_exit_velocity": previous["avg_exit_velocity"],
        "hard_hit_rate": previous["hard_hit_rate"],
        "strikeout_rate": previous["strikeout_rate"],
        "walk_rate": previous["walk_rate"],
        "plate_appearances": previous["plate_appearances"],
        "batted_balls": previous["batted_balls"],
        "recent_ops": previous["ops"],
        "future_ops": future["ops"],
        "ops_change": ops_change,
        "breakout": breakout
    })


# Convert training examples into a DataFrame
training_df = pd.DataFrame(training_rows)

print("\nTraining Data:")
if training_df.empty:
    print("No qualifying 30-game window pairs found.")
else:
    print(training_df.round(3).to_string(index=False))

    training_df.to_csv(
        "data/judge_training_2025.csv",
        index=False
    )

    print(f"\nTraining examples: {len(training_df)}")
    print("Training dataset saved successfully!")
