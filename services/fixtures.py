# Import future annotations to allow using type hints without quotes
from __future__ import annotations

# Import Path for handling file system paths
from pathlib import Path

# Import Optional and Union for type hints
from typing import Optional, Union

# Import pandas for data manipulation
import pandas as pd

# Define required columns for the CSV file
CSV_REQUIRED = [
    "league", "home_team", "away_team",
    "odds_home", "odds_draw", "odds_away", "gameweek"
]

def load_fixtures(csv_path: Union[str, Path], gameweek: Optional[int] = None) -> pd.DataFrame:
    """
    Load fixtures+odds from a CSV file and optionally filter by gameweek.

    Args:
        csv_path: path to odds_gw3.csv (or other CSV in same format).
        gameweek: if provided, only return rows matching this gameweek.

    Returns:
        pd.DataFrame with required columns and basic type conversions.

    Raises:
        FileNotFoundError: if csv_path does not exist.
        ValueError: if required columns are missing.
    """
    # Convert input to Path object for consistent path handling
    csv_path = Path(csv_path)
    # Check if the file exists
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)

    # Check for missing required columns
    missing = [c for c in CSV_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Basic cleaning / type conversions for string columns
    for c in ["league", "home_team", "away_team"]:
        # Convert to string and remove any leading/trailing whitespace
        df[c] = df[c].astype(str).str.strip()

    # Convert odds columns to numeric, coercing errors to NaN
    for c in ["odds_home", "odds_draw", "odds_away"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convert gameweek to numeric, then to nullable integer type
    df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce").astype("Int64")

    # Filter by gameweek if specified
    if gameweek is not None:
        df = df[df["gameweek"] == gameweek]

    # Reset index and return the cleaned DataFrame
    return df.reset_index(drop=True)
