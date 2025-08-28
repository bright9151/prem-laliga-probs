# Import future annotations to allow using type hints without quotes
from __future__ import annotations

# Import Path for handling file system paths
from pathlib import Path

# Import Dash for creating the web application
from dash import Dash

# Import UI layout and callback functions
from ui.layout import build_layout
from ui.callbacks import register_callbacks

# Define project paths
ROOT = Path(__file__).resolve().parent  # Get the root directory of the project
CSV_PATH = ROOT / "data" / "odds_gw3.csv"  # Define path to the CSV data file

# Create Dash app instance
app = Dash(__name__, suppress_callback_exceptions=True)  # Initialize Dash app with callback exception suppression
app.title = "GW3 Probabilities — EPL & LaLiga"  # Set browser tab title
app.layout = build_layout()  # Build the UI layout from layout.py

# Expose Flask server for deployments (gunicorn, etc.)
server = app.server  # Expose the underlying Flask server for production deployment

# Wire callbacks (pass path to CSV)
register_callbacks(app, CSV_PATH)  # Register all interactive callbacks with the CSV path

# Application entry point
if __name__ == "__main__":
    # Quick safety check: warn if CSV missing (helpful for dev)
    if not CSV_PATH.exists():
        print(f"Warning: CSV not found at {CSV_PATH}. Place your odds_gw3.csv there before running.")
    
    # Run the application in debug mode
    app.run(debug=True)
