# services/preprocessor.py
# Import future annotations to allow using type hints without quotes
from __future__ import annotations

# Import Path for handling file system paths
from pathlib import Path

# Import pandas for data manipulation
import pandas as pd

# Import numpy for numerical computations
import numpy as np

# Import Union for type hints that can be multiple types
from typing import Union

# Define required columns for odds data
REQUIRED_ODDS_COLS = ["odds_home", "odds_draw", "odds_away"]


def load_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load CSV into a pandas DataFrame. Accepts string path or Path.
    Raises FileNotFoundError if path doesn't exist.
    """
    # Convert input to Path object for consistent path handling
    path = Path(path)
    # Check if the file exists
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    # Read the CSV file into a DataFrame
    df = pd.read_csv(path)
    return df


def normalize_probs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert decimal odds (odds_home, odds_draw, odds_away) into normalized
    implied probabilities p_home, p_draw, p_away.

    Formula:
        raw_implied = 1 / odds
        overround = sum(raw_implied)
        p = raw_implied / overround

    The function:
    - coerces odds to numeric
    - treats non-positive or non-numeric odds as invalid (resulting probabilities = NaN)
    - returns the input DataFrame (copy) with added columns:
        p_home, p_draw, p_away, overround
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    df = df.copy()

    # Validate presence of required columns
    missing = [c for c in REQUIRED_ODDS_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required odds columns: {missing}")

    # Coerce odds columns to numeric (float), invalid parsing becomes NaN
    odds = df[REQUIRED_ODDS_COLS].apply(pd.to_numeric, errors="coerce")

    # Create a mask for valid rows: all three odds present and > 0
    valid_mask = odds.notna().all(axis=1) & (odds > 0).all(axis=1)

    # Initialize result columns with NaN
    df["p_home"] = np.nan
    df["p_draw"] = np.nan
    df["p_away"] = np.nan
    df["overround"] = np.nan

    # Process valid rows if any exist
    if valid_mask.any():
        # Compute raw implied probabilities (1 / odds) for valid rows
        inv = 1.0 / odds[valid_mask]

        # Sum per-row (overround) - the bookmaker's margin
        s = inv.sum(axis=1)

        # Avoid division by zero (shouldn't happen because we checked >0)
        safe_mask = s > 0

        # Create a DataFrame for normalized probabilities
        normalized = pd.DataFrame(index=inv.index, columns=["p_home", "p_draw", "p_away"], dtype=float)
        # Calculate normalized probabilities for safe rows
        normalized.loc[safe_mask, "p_home"] = (inv.loc[safe_mask, "odds_home"] / s.loc[safe_mask]).astype(float)
        normalized.loc[safe_mask, "p_draw"] = (inv.loc[safe_mask, "odds_draw"] / s.loc[safe_mask]).astype(float)
        normalized.loc[safe_mask, "p_away"] = (inv.loc[safe_mask, "odds_away"] / s.loc[safe_mask]).astype(float)

        # Fill the results back into the main DataFrame
        df.loc[valid_mask, "p_home"] = normalized["p_home"]
        df.loc[valid_mask, "p_draw"] = normalized["p_draw"]
        df.loc[valid_mask, "p_away"] = normalized["p_away"]
        df.loc[valid_mask, "overround"] = s

    # Ensure numeric types for all probability columns
    df["p_home"] = pd.to_numeric(df["p_home"], errors="coerce")
    df["p_draw"] = pd.to_numeric(df["p_draw"], errors="coerce")
    df["p_away"] = pd.to_numeric(df["p_away"], errors="coerce")
    df["overround"] = pd.to_numeric(df["overround"], errors="coerce")

    return df


def preprocess_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Convenience: load CSV from path, ensure 'gameweek' typed, compute probabilities.
    Returns the processed DataFrame with p_home/p_draw/p_away/overround columns.
    """
    # Load the CSV file
    df = load_csv(path)

    # Normalize column names by stripping whitespace
    df.columns = [c.strip() for c in df.columns]

    # Ensure gameweek column is properly typed as nullable integer if present
    if "gameweek" in df.columns:
        df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce").astype("Int64")

    # Compute probabilities from odds
    df = normalize_probs(df)
    # Reset index for cleanliness and return
    return df.reset_index(drop=True)


# Backwards-compatible alias (some callbacks might import compute_probs_from_odds)
def compute_probs_from_odds(path_or_df: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """
    If provided a path (str/Path) it loads and processes CSV; if provided a DataFrame,
    it computes probabilities in-place (on a copy) and returns it.
    """
    # Handle path input
    if isinstance(path_or_df, (str, Path)):
        return preprocess_csv(path_or_df)
    # Handle DataFrame input
    if isinstance(path_or_df, pd.DataFrame):
        return normalize_probs(path_or_df)
    # Raise error for invalid input types
    raise TypeError("compute_probs_from_odds expects a path or a pandas DataFrame")
