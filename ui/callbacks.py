# ui/callbacks.py
# Import future annotations to allow using type hints without quotes
from __future__ import annotations

# Import Path for handling file system paths
from pathlib import Path

# Import type hints for various data types
from typing import Any, Dict, List, Optional

# Import pandas for data manipulation
import pandas as pd

# Import Dash components for building interactive web applications
from dash import Input, Output, State, no_update, html

# Import Plotly for creating interactive visualizations
import plotly.graph_objects as go

# Try to import preprocessor helpers (be flexible about which names exist)
try:
    from services.preprocessor import preprocess_csv, load_csv, normalize_probs  # type: ignore
    HAS_PREPROCESS_CSV = True
except Exception:
    try:
        from services.preprocessor import load_csv, normalize_probs  # type: ignore
        HAS_PREPROCESS_CSV = False
    except Exception as e:
        raise ImportError(
            "services.preprocessor must expose at least 'load_csv' and 'normalize_probs', "
            "or 'preprocess_csv'. Error: " + str(e)
        )

# Mapping for league dropdown values -> CSV labels
_LEAGUE_VALUE_TO_LABEL = {
    "EPL": "Premier League",
    "premier_league": "Premier League",
    "Premier League": "Premier League",
    "LaLiga": "LaLiga",
    "la_liga": "LaLiga",
    "La Liga": "LaLiga",
}


def _safe_get_league_label(value: Optional[str]) -> Optional[str]:
    # Safely get league label from mapping, return original value if not found
    if value is None:
        return None
    return _LEAGUE_VALUE_TO_LABEL.get(value, value)


def _compute_probs_from_odds(df: pd.DataFrame) -> pd.DataFrame:
    # Compute probabilities from odds if not already present
    df = df.copy()
    required = ["odds_home", "odds_draw", "odds_away"]
    if not all(col in df.columns for col in required):
        # If required columns are missing, add empty probability columns
        df["p_home"] = pd.NA
        df["p_draw"] = pd.NA
        df["p_away"] = pd.NA
        return df

    # Convert odds to numeric
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Calculate implied probabilities (1/odds)
    inv = 1.0 / df[required]
    # Calculate overround (sum of implied probabilities)
    s = inv.sum(axis=1)
    # Normalize probabilities and handle division by zero
    with pd.option_context("mode.use_inf_as_na", True):
        df.loc[:, "p_home"] = (inv["odds_home"] / s).where(s > 0, pd.NA)
        df.loc[:, "p_draw"] = (inv["odds_draw"] / s).where(s > 0, pd.NA)
        df.loc[:, "p_away"] = (inv["odds_away"] / s).where(s > 0, pd.NA)

    return df


def _ensure_prob_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure probability columns exist and are properly formatted
    df = df.copy()
    # Check if standard probability columns exist
    if {"p_home", "p_draw", "p_away"}.issubset(df.columns):
        df["p_home"] = pd.to_numeric(df["p_home"], errors="coerce")
        df["p_draw"] = pd.to_numeric(df["p_draw"], errors="coerce")
        df["p_away"] = pd.to_numeric(df["p_away"], errors="coerce")
        return df

    # Check if alternative probability column names exist
    if {"Home_Prob", "Draw_Prob", "Away_Prob"}.issubset(df.columns):
        df["p_home"] = pd.to_numeric(df["Home_Prob"], errors="coerce")
        df["p_draw"] = pd.to_numeric(df["Draw_Prob"], errors="coerce")
        df["p_away"] = pd.to_numeric(df["Away_Prob"], errors="coerce")
        return df

    # If no probability columns found, compute from odds
    df = _compute_probs_from_odds(df)
    return df


def _format_pct(x: Any) -> str:
    # Format a decimal value as a percentage string
    try:
        val = float(x)
        return f"{val * 100:.1f}%"
    except Exception:
        return "N/A"


def _build_table_html(df: pd.DataFrame) -> html.Div:
    """Build styled HTML table with colored probability cells."""
    # Create table header with styling
    header = html.Tr(
        [html.Th("Home"), html.Th("Away"), html.Th("Home %"), html.Th("Draw %"), html.Th("Away %")],
        style={"textAlign": "center", "backgroundColor": "#2c3e50", "color": "white", "fontWeight": "bold"}
    )

    # Create table rows
    rows = []
    for _, r in df.iterrows():
        rows.append(
            html.Tr(
                [
                    html.Td(r.get("home_team", ""), style={"padding": "6px", "fontWeight": "600"}),
                    html.Td(r.get("away_team", ""), style={"padding": "6px", "fontWeight": "600"}),
                    html.Td(_format_pct(r.get("p_home")), style={"color": "green", "fontWeight": "bold"}),
                    html.Td(_format_pct(r.get("p_draw")), style={"color": "grey", "fontWeight": "bold"}),
                    html.Td(_format_pct(r.get("p_away")), style={"color": "red", "fontWeight": "bold"}),
                ],
                # Alternate row background colors
                style={"backgroundColor": "#f9f9f9"} if _ % 2 == 0 else {"backgroundColor": "#ffffff"},
            )
        )

    # Create the complete table
    table = html.Table(
        [html.Thead(header), html.Tbody(rows)],
        style={
            "width": "100%",
            "borderCollapse": "collapse",
            "border": "1px solid #ddd",
            "marginTop": "15px",
            "boxShadow": "0 2px 6px rgba(0,0,0,0.1)",
        }
    )
    return html.Div(table, style={"overflowX": "auto"})


def _figure_for_row(row: pd.Series) -> go.Figure:
    """Build a bar chart with consistent colors (green=home, grey=draw, red=away)."""
    # Get team names
    home = str(row.get("home_team", "Home"))
    away = str(row.get("away_team", "Away"))

    # Get probabilities
    try:
        ph = float(row.get("p_home") or 0.0)
        pd_ = float(row.get("p_draw") or 0.0)
        pa = float(row.get("p_away") or 0.0)
    except Exception:
        ph = pd_ = pa = 0.0

    # Prepare data for bar chart
    x = [f"{home}\nWin", "Draw", f"{away}\nWin"]
    y = [ph, pd_, pa]
    text = [f"{v*100:.1f}%" for v in y]
    colors = ["green", "grey", "red"]

    # Create bar chart
    fig = go.Figure(data=[go.Bar(x=x, y=y, text=text, textposition="outside", marker_color=colors)])
    # Format y-axis as percentage
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    # Update layout for better appearance
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=10),
        xaxis_title=None,
        yaxis_title="Probability",
        plot_bgcolor="white",
    )
    return fig


def register_callbacks(app, csv_path: Path):
    """Register callbacks for the app."""
    # Load and preprocess data
    csv_path = Path(csv_path)
    try:
        if HAS_PREPROCESS_CSV:
            df_all = preprocess_csv(csv_path)
        else:
            df_all = load_csv(csv_path)
            try:
                maybe = normalize_probs(df_all)
                if isinstance(maybe, pd.DataFrame):
                    df_all = maybe
                else:
                    df_all = df_all.apply(normalize_probs, axis=1)
            except Exception:
                pass
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at {csv_path}.")
    except Exception as e:
        raise RuntimeError(f"Failed to load or preprocess CSV: {e}")

    # Ensure gameweek column is properly formatted
    if "gameweek" in df_all.columns:
        df_all["gameweek"] = pd.to_numeric(df_all["gameweek"], errors="coerce").astype("Int64")

    # Callback to update gameweek dropdown based on league selection
    @app.callback(
        Output("gw-dropdown", "options"),
        Input("league-dropdown", "value"),
        prevent_initial_call=False,
    )
    def update_gameweek_options(league_value: Optional[str]):
        # Get available gameweeks for selected league
        league_label = _safe_get_league_label(league_value)
        if league_label:
            weeks = df_all.loc[df_all["league"] == league_label, "gameweek"].dropna().unique()
        else:
            weeks = df_all["gameweek"].dropna().unique()
        # Sort gameweeks numerically
        try:
            weeks_sorted = sorted(int(w) for w in weeks)
        except Exception:
            weeks_sorted = sorted(weeks)
        # Format options for dropdown
        return [{"label": f"Gameweek {w}", "value": w} for w in weeks_sorted]

    # Callback to update table and chart based on league and gameweek selection
    @app.callback(
        Output("probs-table", "children"),
        Output("probs-chart", "figure"),
        Input("league-dropdown", "value"),
        Input("gw-dropdown", "value"),
        prevent_initial_call=False,
    )
    def update_view(league_value: Optional[str], gw_value: Optional[int]):
        # Filter data based on selections
        df = df_all.copy()
        if league_value:
            league_label = _safe_get_league_label(league_value)
            df = df[df["league"] == league_label]
        if gw_value is not None and "gameweek" in df.columns:
            try:
                gw_int = int(gw_value)
                df = df[df["gameweek"] == gw_int]
            except Exception:
                pass

        # Handle empty results
        if df.empty:
            return html.Div("No matches found for selection."), go.Figure()

        # Ensure probability columns exist
        df = _ensure_prob_cols(df)
        # Format percentages for display
        df["home_pct"] = df["p_home"].apply(_format_pct)
        df["draw_pct"] = df["p_draw"].apply(_format_pct)
        df["away_pct"] = df["p_away"].apply(_format_pct)

        # Build table and chart
        table_div = _build_table_html(df)
        first_row = df.iloc[0]
        fig = _figure_for_row(first_row)
        return table_div, fig

    # Callback to handle CSV download
    @app.callback(
        Output("download-data", "data"),
        Input("download-btn", "n_clicks"),
        State("league-dropdown", "value"),
        State("gw-dropdown", "value"),
        prevent_initial_call=True,
    )
    def download_csv(n_clicks: int, league_value: Optional[str], gw_value: Optional[int]):
        # Only trigger on button click
        if not n_clicks:
            return no_update

        # Filter data based on selections
        df = df_all.copy()
        if league_value:
            league_label = _safe_get_league_label(league_value)
            df = df[df["league"] == league_label]
        if gw_value is not None and "gameweek" in df.columns:
            try:
                gw_int = int(gw_value)
                df = df[df["gameweek"] == gw_int]
            except Exception:
                pass
        if df.empty:
            return no_update

        # Ensure probability columns exist
        df = _ensure_prob_cols(df)
        # Define export columns
        export_cols = ["league", "home_team", "away_team", "odds_home", "odds_draw", "odds_away",
                       "p_home", "p_draw", "p_away", "gameweek"]
        # Add missing columns with NA values
        for c in export_cols:
            if c not in df.columns:
                df[c] = pd.NA
        # Convert to CSV and return for download
        csv_str = df[export_cols].to_csv(index=False)
        return dict(content=csv_str, filename="gw3_probabilities.csv")
