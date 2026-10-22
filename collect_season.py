from pathlib import Path
from datetime import date, timedelta
import pandas as pd
from pybaseball import statcast, cache

# Enable caching for Statcast requests
cache.enable()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--season", type=int, default=2025)
args = parser.parse_args()

SEASON = args.season

SEASON_DATES = {
    2023: (date(2023, 3, 30), date(2023, 10, 1)),
    2024: (date(2024, 3, 20), date(2024, 9, 30)),
    2025: (date(2025, 3, 27), date(2025, 9, 28)),
}

START_DATE, END_DATE = SEASON_DATES[SEASON]

# Store weekly data in separate files
DATA_DIR = Path(f"data/statcast_{SEASON}")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def collect_season():
    current_date = START_DATE

    while current_date <= END_DATE:
        # Create a seven-day date range
        chunk_end = min(
            current_date + timedelta(days=6),
            END_DATE
        )

        filename = DATA_DIR / (
            f"{current_date}_{chunk_end}.csv"
        )

        # Skip previously downloaded chunks
        if filename.exists():
            print(f"Skipping existing file: {filename.name}")
            current_date = chunk_end + timedelta(days=1)
            continue

        print(f"\nDownloading {current_date} to {chunk_end}")

        try:
            data = statcast(
                start_dt=str(current_date),
                end_dt=str(chunk_end),
                parallel=False
            )

            # Keep regular-season data only
            if not data.empty:
                data = data[data["game_type"] == "R"]

            # Save the weekly dataset
            data.to_csv(filename, index=False)

            print(f"Saved {len(data)} pitches")

        except Exception as error:
            print(f"Download failed: {error}")
            print("Run the script again to retry this chunk.")
            raise

        current_date = chunk_end + timedelta(days=1)

    print("\nSeason collection complete!")


if __name__ == "__main__":
    collect_season()
