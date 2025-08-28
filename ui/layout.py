# ui/layout.py
# Import Dash components for building the web layout
from dash import dcc, html

def build_layout():
    # Create the main layout container with a gradient background
    return html.Div(
        style={
            "background": "linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%)",  # Gradient background
            "minHeight": "100vh",  # Minimum height of 100% of viewport height
            "padding": "30px",  # Padding around the content
            "fontFamily": "Arial, sans-serif",  # Font family for text
        },
        children=[
            # Main title/header of the dashboard
            html.H1(
                "⚽ Premier League & La Liga Odds Dashboard",  # Title with soccer emoji
                style={
                    "textAlign": "center",  # Center align text
                    "color": "#ffffff",  # White text color
                    "fontSize": "40px",  # Large font size
                    "marginBottom": "40px",  # Bottom margin
                    "textShadow": "2px 2px 4px rgba(0,0,0,0.4)",  # Text shadow for better readability
                },
            ),

            # League + GW dropdowns container
            html.Div(
                style={
                    "display": "flex",  # Use flexbox for layout
                    "justifyContent": "center",  # Center items horizontally
                    "gap": "20px",  # Space between items
                    "marginBottom": "30px",  # Bottom margin
                },
                children=[
                    # League selection dropdown
                    dcc.Dropdown(
                        id="league-dropdown",  # Unique identifier for callback targeting
                        options=[  # Available options in the dropdown
                            {"label": "Premier League", "value": "premier_league"},
                            {"label": "La Liga", "value": "la_liga"}
                        ],
                        value="premier_league",  # Default selected value
                        style={
                            "width": "280px",  # Fixed width
                            "padding": "8px",  # Internal padding
                            "borderRadius": "12px",  # Rounded corners
                            "backgroundColor": "#ffffff",  # White background
                            "boxShadow": "0 4px 10px rgba(0,0,0,0.2)",  # Shadow effect
                        },
                    ),
                    # Gameweek selection dropdown
                    dcc.Dropdown(
                        id="gw-dropdown",  # Unique identifier for callback targeting
                        options=[],  # Options will be populated by callback
                        placeholder="Select Gameweek",  # Placeholder text
                        style={
                            "width": "220px",  # Fixed width
                            "padding": "8px",  # Internal padding
                            "borderRadius": "12px",  # Rounded corners
                            "backgroundColor": "#ffffff",  # White background
                            "boxShadow": "0 4px 10px rgba(0,0,0,0.2)",  # Shadow effect
                        },
                    ),
                ],
            ),

            # Table card container
            html.Div(
                id="probs-table",  # Unique identifier for callback targeting
                style={
                    "backgroundColor": "#ffffff",  # White background
                    "padding": "20px",  # Internal padding
                    "borderRadius": "15px",  # Rounded corners
                    "boxShadow": "0 6px 15px rgba(0,0,0,0.25)",  # Shadow effect
                    "marginBottom": "30px",  # Bottom margin
                    "border": "3px solid #74b9ff",  # Blue border
                },
            ),

            # Chart card container
            html.Div(
                dcc.Graph(id="probs-chart"),  # Graph component for displaying visualizations
                style={
                    "backgroundColor": "#ffffff",  # White background
                    "padding": "20px",  # Internal padding
                    "borderRadius": "15px",  # Rounded corners
                    "boxShadow": "0 6px 15px rgba(0,0,0,0.25)",  # Shadow effect
                    "marginBottom": "30px",  # Bottom margin
                    "border": "3px solid #a29bfe",  # Purple border
                },
            ),

            # Download button container
            html.Div(
                style={"textAlign": "center", "marginTop": "20px"},  # Center align and top margin
                children=[
                    # Download button
                    html.Button(
                        "⬇ Download Data",  # Button text with download emoji
                        id="download-btn",  # Unique identifier for callback targeting
                        n_clicks=0,  # Initial click count
                        style={
                            "backgroundColor": "#e17055",  # Orange background
                            "color": "white",  # White text
                            "border": "none",  # No border
                            "padding": "12px 24px",  # Internal padding
                            "borderRadius": "12px",  # Rounded corners
                            "cursor": "pointer",  # Pointer cursor on hover
                            "fontSize": "18px",  # Font size
                            "boxShadow": "0 4px 10px rgba(0,0,0,0.3)",  # Shadow effect
                        },
                    ),
                    # Hidden download component that handles file downloads
                    dcc.Download(id="download-data"),
                ],
            ),
        ],
    )
