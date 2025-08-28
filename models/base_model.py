# Import future annotations to allow using type hints without quotes (for forward references)
from __future__ import annotations

# Import ABC (Abstract Base Class) and abstractmethod for creating abstract classes/methods
from abc import ABC, abstractmethod

# Import pandas library for data manipulation, typically used with DataFrames
import pandas as pd

# Define an abstract base class for probability models using ABC
class BaseProbabilityModel(ABC):
    """
    Abstract interface for probability models.

    Subclasses must implement predict_proba(df), returning a DataFrame that
    includes columns: p_home, p_draw, p_away (all between 0 and 1).
    """
    
    # Declare an abstract method that must be implemented by subclasses
    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        # Raise NotImplementedError if subclass doesn't implement this method
        raise NotImplementedError
