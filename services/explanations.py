# Import Dict for type hinting
from typing import Dict

def explain_match(row: Dict) -> str:
    """
    Generate a simple reasoning string for why the probabilities look that way.
    Uses league, teams, and implied probabilities.
    """

    # Extract home team name from row, default to "Home" if not found
    home = row.get("home_team", "Home")
    # Extract away team name from row, default to "Away" if not found
    away = row.get("away_team", "Away")
    # Extract league name from row, default to "Match" if not found
    league = row.get("league", "Match")

    # Extract probability of home win, default to 0 if not found
    p_home = row.get("p_home", 0)
    # Extract probability of draw, default to 0 if not found
    p_draw = row.get("p_draw", 0)
    # Extract probability of away win, default to 0 if not found
    p_away = row.get("p_away", 0)

    # Create a dictionary mapping outcome names to their probabilities
    probs = {"Home win": p_home, "Draw": p_draw, "Away win": p_away}
    # Find the outcome with the highest probability
    favorite = max(probs, key=probs.get)
    # Get the probability value of the favorite outcome
    fav_prob = probs[favorite]

    # Start building the explanation string
    explanation = (
        f"In the {league} clash between {home} and {away}, "
        f"the model favors **{favorite}** with a probability of {fav_prob:.1%}. "
    )

    # Add context based on which outcome is favored
    if favorite == "Home win":
        explanation += f"{home} likely benefits from home advantage."
    elif favorite == "Away win":
        explanation += f"{away} may be in stronger form or have better squad depth."
    else:
        explanation += "Both teams seem evenly matched, hence a high draw probability."

    # Return the complete explanation
    return explanation
