
from pathlib import Path
import pandas as pd

# --------------------------------------------------------
#                    CONFIGURATION
# --------------------------------------------------------

SEASON = 2025
WINDOW_SIZE = 30
STEP_SIZE = 10
MIN_PLATE_APPEARANCES = 75
MIN_BATTED_BALLS = 20

DATA_DIR = Path(f"data/statcast_{SEASON}")
OUTPUT_FILE = Path(f"data/training_{SEASON}.csv")

# Only load columns needed for our first model
COLUMNS = [
    "game_pk", "game_date", "game_type",
    "batter", "at_bat_number", "pitch_number",
    "events", "type", "launch_speed"
]

# --------------------------------------------------------
#                    LOAD DATA
# --------------------------------------------------------

def load_season():
    files = sorted(DATA_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError("No Statcast files found.")

    print(f"Loading {len(files)} weekly datasets...")

    frames = [
        pd.read_csv(file, usecols=COLUMNS, low_memory=False)
        for file in files
    ]

    data = pd.concat(frames, ignore_index=True)

    # Remove duplicate pitch records
    before = len(data)

    data = data.drop_duplicates(
        subset=["game_pk", "at_bat_number", "pitch_number"]
    )

    print(f"Duplicate pitches removed: {before - len(data)}")

    # Ensure numerical measurements
    data["launch_speed"] = pd.to_numeric(
        data["launch_speed"], errors="coerce"
    )

    data = data[data["game_type"] == "R"].copy()
    data["game_date"] = pd.to_datetime(data["game_date"])

    print(f"Total pitches loaded: {len(data):,}")
    print(f"Unique batters: {data['batter'].nunique()}")

    return data


# --------------------------------------------------------
#                    CALCULATE OPS
# --------------------------------------------------------

HIT_BASES = {
    "single": 1,
    "double": 2,
    "triple": 3,
    "home_run": 4
}

NON_AB_EVENTS = {
    "walk", "intent_walk", "hit_by_pitch",
    "sac_fly", "sac_fly_double_play",
    "sac_bunt", "sac_bunt_double_play",
    "catcher_interf"
}

def calculate_ops(pa):
    events = pa["events"]

    hits = events.isin(HIT_BASES).sum()
    total_bases = events.map(HIT_BASES).fillna(0).sum()

    walks = events.isin(["walk", "intent_walk"]).sum()
    hbp = (events == "hit_by_pitch").sum()
    sac_flies = events.isin([
        "sac_fly", "sac_fly_double_play"
    ]).sum()

    at_bats = (~events.isin(NON_AB_EVENTS)).sum()
    denominator = at_bats + walks + hbp + sac_flies

    if at_bats == 0 or denominator == 0:
        return None

    obp = (hits + walks + hbp) / denominator
    slg = total_bases / at_bats

    return obp + slg


# --------------------------------------------------------
#                 CALCULATE FEATURES
# --------------------------------------------------------

def calculate_features(pitches):
    pa = pitches[pitches["events"].notna()]

    batted = pitches[
        (pitches["type"] == "X") &
        (pitches["launch_speed"].notna())
    ]

    if (
        len(pa) < MIN_PLATE_APPEARANCES or
        len(batted) < MIN_BATTED_BALLS
    ):
        return None

    ops = calculate_ops(pa)

    if ops is None:
        return None

    return {
        "avg_exit_velocity": batted["launch_speed"].mean(),
        "hard_hit_rate": (batted["launch_speed"] >= 95).mean(),
        "strikeout_rate": pa["events"].isin([
            "strikeout", "strikeout_double_play"
        ]).mean(),
        "walk_rate": pa["events"].isin([
            "walk", "intent_walk"
        ]).mean(),
        "plate_appearances": len(pa),
        "batted_balls": len(batted),
        "ops": ops
    }


# --------------------------------------------------------
#                 GENERATE TRAINING DATA
# --------------------------------------------------------

def build_training_data(data):
    training_rows = []

    # Process each hitter separately
    for batter, player_data in data.groupby("batter"):
        player_data = player_data.sort_values([
            "game_date", "game_pk",
            "at_bat_number", "pitch_number"
        ])

        # Only include games with completed plate appearances
        games = (
            player_data[player_data["events"].notna()]
            [["game_date", "game_pk"]]
            .drop_duplicates()
            .sort_values(["game_date", "game_pk"])
            ["game_pk"]
            .tolist()
        )

        if len(games) < 2 * WINDOW_SIZE:
            continue

        # Create 30-game input and future windows
        for start in range(
            0,
            len(games) - 2 * WINDOW_SIZE + 1,
            STEP_SIZE
        ):
            previous_games = games[start:start + WINDOW_SIZE]
            future_games = games[
                start + WINDOW_SIZE:start + 2 * WINDOW_SIZE
            ]

            previous_data = player_data[
                player_data["game_pk"].isin(previous_games)
            ]

            future_data = player_data[
                player_data["game_pk"].isin(future_games)
            ]

            previous = calculate_features(previous_data)
            future = calculate_features(future_data)

            if previous is None or future is None:
                continue

            ops_change = future["ops"] - previous["ops"]

            training_rows.append({
                "batter": batter,
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
                "breakout": int(ops_change >= 0.075)
            })

    return pd.DataFrame(training_rows)


# --------------------------------------------------------
#                         MAIN
# --------------------------------------------------------

if __name__ == "__main__":
    data = load_season()

    print("\nGenerating training examples...")
    training_df = build_training_data(data)

    if training_df.empty:
        print("No qualifying training examples found.")
    else:
        print(f"\nTraining examples: {len(training_df):,}")
        print(f"Unique hitters: {training_df['batter'].nunique()}")

        print("\nBreakout label distribution:")
        print(training_df["breakout"].value_counts())

        print("\nFirst 10 examples:")
        print(training_df.head(10).round(3).to_string(index=False))

        # Save the completed training dataset
        training_df.to_csv(OUTPUT_FILE, index=False)

        print(f"\nSaved to: {OUTPUT_FILE}")
