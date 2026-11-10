"""Data loading.

Dataquest NBA MVP dataset (1991-2021), scraped from Basketball-Reference.
Pinned to a fixed commit so everyone gets the same data.
"""
import pandas as pd

DATA_URL = ("https://raw.githubusercontent.com/dataquestio/project-walkthroughs/"
            "571e6565edb7330c65a1dcb35033dac5856ffd6d/mvp/player_mvp_stats.csv")


def load_data() -> pd.DataFrame:
    """Return the raw player-season table (one row per player per season)."""
    return pd.read_csv(DATA_URL)
