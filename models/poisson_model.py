# Import future annotations to allow using type hints without quotes
from __future__ import annotations

# Import pandas for data manipulation
import pandas as pd

# Import numpy for numerical computations
import numpy as np

# Import math for mathematical functions
import math

# Import type hints for optional parameters and dictionaries
from typing import Optional, Dict

# Import the base class that this class inherits from
from .base_model import BaseProbabilityModel

class PoissonModel(BaseProbabilityModel):
    """
    Simple Poisson-based model to estimate match outcome probabilities.

    How it works (simple, explainable approach):
    - Call fit(matches_df) with historical matches containing:
        columns = ['home_team','away_team','home_goals','away_goals']
    - The fitter computes:
        * league average goals at home (mu_home) and away (mu_away)
        * per-team attack/defense multipliers for home/away
    - predict_proba(df) expects df with 'home_team' and 'away_team' columns
      and returns p_home, p_draw, p_away computed by summing Poisson scoreline probs.

    Notes:
    - This is a lightweight, explainable model (not a full statistical model).
    - If the model has not been fitted, predict_proba will return NaNs and a warning.
    """

    def __init__(self, max_goals: int = 6):
        # Initialize the model with maximum number of goals to consider
        self.max_goals = int(max_goals)
        # Flag to track if the model has been fitted
        self.fitted = False
        # League average goals at home
        self.mu_home: Optional[float] = None
        # League average goals away
        self.mu_away: Optional[float] = None
        # Dictionary to store team statistics
        self.team_stats: Dict[str, Dict[str, float]] = {}

    def fit(self, matches_df: pd.DataFrame) -> None:
        """
        Fit the model using historical matches DataFrame. Required columns:
        'home_team','away_team','home_goals','away_goals'
        """
        # Define required columns
        required = {'home_team','away_team','home_goals','away_goals'}
        # Check if all required columns are present
        if not required.issubset(set(matches_df.columns)):
            raise ValueError(f"matches_df must include columns: {required}")

        # Create a copy of the input DataFrame
        df = matches_df.copy()
        # Compute league average goals at home
        self.mu_home = float(df['home_goals'].mean())
        # Compute league average goals away
        self.mu_away = float(df['away_goals'].mean())

        # Get unique teams from both home and away columns
        teams = pd.unique(df[['home_team','away_team']].values.ravel('K'))

        # Initialize dictionary to store team statistics
        stats = {}
        # Calculate statistics for each team
        for t in teams:
            # Filter home games for the current team
            home_mask = df['home_team'] == t
            # Filter away games for the current team
            away_mask = df['away_team'] == t

            # Calculate average goals scored at home
            gf_home = df.loc[home_mask, 'home_goals'].mean() if home_mask.any() else np.nan
            # Calculate average goals conceded at home
            ga_home = df.loc[home_mask, 'away_goals'].mean() if home_mask.any() else np.nan

            # Calculate average goals scored away
            gf_away = df.loc[away_mask, 'away_goals'].mean() if away_mask.any() else np.nan
            # Calculate average goals conceded away
            ga_away = df.loc[away_mask, 'home_goals'].mean() if away_mask.any() else np.nan

            # Fallback to league averages if insufficient data
            if pd.isna(gf_home): gf_home = self.mu_home
            if pd.isna(ga_home): ga_home = self.mu_away
            if pd.isna(gf_away): gf_away = self.mu_away
            if pd.isna(ga_away): ga_away = self.mu_home

            # Calculate attack and defense multipliers
            # Avoid division by zero
            attack_home = (gf_home / self.mu_home) if self.mu_home > 0 else 1.0
            defense_home = (ga_home / self.mu_away) if self.mu_away > 0 else 1.0
            attack_away = (gf_away / self.mu_away) if self.mu_away > 0 else 1.0
            defense_away = (ga_away / self.mu_home) if self.mu_home > 0 else 1.0

            # Store all statistics for the team
            stats[t] = {
                'gf_home': gf_home, 'ga_home': ga_home,
                'gf_away': gf_away, 'ga_away': ga_away,
                'attack_home': attack_home, 'defense_home': defense_home,
                'attack_away': attack_away, 'defense_away': defense_away
            }

        # Store team statistics and mark model as fitted
        self.team_stats = stats
        self.fitted = True

    @staticmethod
    def _poisson_pmf(k: int, lamb: float) -> float:
        # Poisson Probability Mass Function: e^{-lambda} lambda^k / k!
        if lamb < 0:
            return 0.0
        # guard against large factorials when k is big (k will be small)
        return math.exp(-lamb) * (lamb ** k) / math.factorial(k)

    def _score_matrix(self, lambda_home: float, lambda_away: float) -> np.ndarray:
        """
        Build score probability matrix P[i,j] = P(home scores i) * P(away scores j)
        for i,j = 0..max_goals
        """
        # Size of the matrix (0 to max_goals inclusive)
        size = self.max_goals + 1
        # Calculate probability of home team scoring i goals
        p_home_goals = [self._poisson_pmf(i, lambda_home) for i in range(size)]
        # Calculate probability of away team scoring j goals
        p_away_goals = [self._poisson_pmf(j, lambda_away) for j in range(size)]
        # Create matrix of joint probabilities
        mat = np.outer(p_home_goals, p_away_goals)
        # probability mass outside considered goals (tails) is ignored; max_goals should be >=5 typically
        return mat

    def _match_expectations(self, home: str, away: str) -> (float, float):
        """
        Compute expected goals (lambda_home, lambda_away) for a match home vs away.
        Uses formulae:
          lambda_home = mu_home * attack_home(home) * defense_away(away)
          lambda_away = mu_away * attack_away(away) * defense_home(home)
        Falls back to league mu's and multipliers of 1.0 if team not seen.
        """
        # Check if model has been fitted
        if not self.fitted:
            raise RuntimeError("PoissonModel must be fitted before computing expectations.")

        # get team stats or default multipliers = 1.0
        h_stat = self.team_stats.get(home, None)
        a_stat = self.team_stats.get(away, None)

        # Get home team's home attack multiplier
        att_home = h_stat['attack_home'] if h_stat is not None else 1.0
        # Get away team's away defense multiplier
        def_away = a_stat['defense_away'] if a_stat is not None else 1.0
        # Get away team's away attack multiplier
        att_away = a_stat['attack_away'] if a_stat is not None else 1.0
        # Get home team's home defense multiplier
        def_home = h_stat['defense_home'] if h_stat is not None else 1.0

        # Calculate expected goals for home team
        lambda_home = float(self.mu_home * att_home * def_away)
        # Calculate expected goals for away team
        lambda_away = float(self.mu_away * att_away * def_home)
        return lambda_home, lambda_away

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Given df with columns 'home_team' and 'away_team', return df with added:
        p_home, p_draw, p_away, lambda_home, lambda_away
        """
        # Check if model has been fitted
        if not self.fitted:
            # return same df with NaNs (safe behavior)
            out = df.copy().reset_index(drop=True)
            out['p_home'] = np.nan
            out['p_draw'] = np.nan
            out['p_away'] = np.nan
            out['lambda_home'] = np.nan
            out['lambda_away'] = np.nan
            return out

        # Initialize list to store output rows
        out_rows = []
        # Iterate through each row in the input DataFrame
        for _, row in df.reset_index(drop=True).iterrows():
            # Get home and away team names
            home = str(row.get('home_team'))
            away = str(row.get('away_team'))

            try:
                # Calculate expected goals for the match
                lambda_h, lambda_a = self._match_expectations(home, away)
            except Exception:
                # Fall back to league averages if calculation fails
                lambda_h, lambda_a = float(self.mu_home), float(self.mu_away)

            # Create score probability matrix
            mat = self._score_matrix(lambda_h, lambda_a)
            # P(home win) = sum_{i>j} mat[i,j]
            p_home = float(np.tril(mat, -1).sum())  # lower triangle (i>j)
            # P(draw) = sum diag
            p_draw = float(np.trace(mat))
            # P(away win) = sum_{i<j} mat[i,j]
            p_away = float(np.triu(mat, 1).sum())  # upper triangle (i<j)

            # numerical safety: normalize if tiny float inaccuracies
            s = p_home + p_draw + p_away
            if s > 0:
                # Normalize probabilities to sum to 1
                p_home, p_draw, p_away = p_home/s, p_draw/s, p_away/s
            else:
                # Set to NaN if probabilities don't sum to a positive value
                p_home = p_draw = p_away = np.nan

            # Add row to output
            out_rows.append({
                **row.to_dict(),  # Include all original row data
                'p_home': p_home,
                'p_draw': p_draw,
                'p_away': p_away,
                'lambda_home': lambda_h,
                'lambda_away': lambda_a
            })

        # Convert list of rows to DataFrame
        return pd.DataFrame(out_rows)
