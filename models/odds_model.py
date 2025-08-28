# Import future annotations to allow using type hints without quotes
from __future__ import annotations

# Import pandas for data manipulation and analysis
import pandas as pd

# Import numpy for numerical computations
import numpy as np

# Import Path for handling file system paths
from pathlib import Path

# Import Optional for type hints that can be None
from typing import Optional

# Import the base class that this class inherits from
from .base_model import BaseProbabilityModel

# Define required columns for the input data
REQUIRED_COLS = [
    "league", "home_team", "away_team",
    "odds_home", "odds_draw", "odds_away", "gameweek"
]

class OddsModel(BaseProbabilityModel):
    """
    Convert 1X2 decimal odds into normalized probabilities.

    - Can read from a provided CSV path via load_and_predict()
    - Or accept a DataFrame into predict_proba(df).
    """

    def __init__(self, csv_path: Optional[Path] = None):
        # Initialize the model with an optional path to a CSV file
        # Convert to Path object if provided, otherwise set to None
        self.csv_path = Path(csv_path) if csv_path is not None else None

    @staticmethod
    def _clean_types(df: pd.DataFrame) -> pd.DataFrame:
        # Clean and standardize data types in the DataFrame
        # Trim strings and enforce floats/ints
        for c in ["league", "home_team", "away_team"]:
            if c in df.columns:
                # Convert to string and remove any leading/trailing whitespace
                df[c] = df[c].astype(str).str.strip()
        for c in ["odds_home", "odds_draw", "odds_away"]:
            if c in df.columns:
                # Convert to numeric, coercing any errors to NaN
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "gameweek" in df.columns:
            # Convert to numeric, then to nullable integer type
            df["gameweek"] = pd.to_numeric(df["gameweek"], errors="coerce").astype("Int64")
        return df

    @staticmethod
    def _from_odds_to_probs(row: pd.Series) -> pd.Series:
        """
        Given a row with decimal odds (odds_home, odds_draw, odds_away),
        return normalized probabilities p_home, p_draw, p_away and the overround.
        """
        # Extract odds values from the row
        oh = row.get("odds_home")
        od = row.get("odds_draw")
        oa = row.get("odds_away")

        # Validate - check for missing values
        if pd.isna(oh) or pd.isna(od) or pd.isna(oa):
            return pd.Series({"p_home": np.nan, "p_draw": np.nan, "p_away": np.nan, "overround": np.nan})
        try:
            # Convert to floats
            oh_f, od_f, oa_f = float(oh), float(od), float(oa)
        except Exception:
            # Return NaN series if conversion fails
            return pd.Series({"p_home": np.nan, "p_draw": np.nan, "p_away": np.nan, "overround": np.nan})
        # Check for positive odds
        if oh_f <= 0 or od_f <= 0 or oa_f <= 0:
            return pd.Series({"p_home": np.nan, "p_draw": np.nan, "p_away": np.nan, "overround": np.nan})

        # Calculate implied probabilities (1/odds)
        inv = np.array([1.0/oh_f, 1.0/od_f, 1.0/oa_f], dtype=float)
        # Sum of implied probabilities (overround)
        s = inv.sum()
        if s <= 0:
            return pd.Series({"p_home": np.nan, "p_draw": np.nan, "p_away": np.nan, "overround": np.nan})

        # Normalize probabilities to sum to 1
        probs = inv / s
        return pd.Series({
            "p_home": float(probs[0]),
            "p_draw": float(probs[1]),
            "p_away": float(probs[2]),
            "overround": float(s)  # Bookmaker's margin
        })

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        # Implementation of the abstract method from BaseProbabilityModel
        # Ensure required columns exist (only check presence)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Clean data types and create a copy
        df = self._clean_types(df).copy()
        # Apply odds-to-probability conversion to each row
        probs = df.apply(self._from_odds_to_probs, axis=1)
        # Combine original data with calculated probabilities
        out = pd.concat([df.reset_index(drop=True), probs.reset_index(drop=True)], axis=1)

        # Calculate favourite team based on highest probability
        def compute_favourite(row: pd.Series) -> str:
            trio = [row.get("p_home"), row.get("p_draw"), row.get("p_away")]
            if any(pd.isna(trio)):
                return "unknown"
            labels = ["home", "draw", "away"]
            # Return label with highest probability
            return labels[int(np.nanargmax(trio))]

        # Apply favourite calculation to each row
        out["favourite"] = out.apply(compute_favourite, axis=1)
        # Calculate edge (difference between highest and lowest probability)
        out["edge"] = (out[["p_home", "p_draw", "p_away"]].max(axis=1) - out[["p_home", "p_draw", "p_away"]].min(axis=1)).astype(float)
        return out

    def load_and_predict(self) -> pd.DataFrame:
        # Load data from CSV and run prediction
        if self.csv_path is None:
            raise ValueError("csv_path was not provided to OddsModel.")
        # Read CSV file
        df = pd.read_csv(self.csv_path)
        # Run prediction on loaded data
        return self.predict_proba(df)
