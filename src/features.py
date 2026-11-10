"""Label, features, and the time-based train/test split."""
import pandas as pd

TRAIN_END = 2012     # train on seasons <= 2012
TEST_START = 2013    # test on seasons >= 2013

# Rank within each season (1 = best): an MVP is whoever stands out vs. that year's league.
FEATURES = ["ppg_rank", "win_rank"]


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """mvp = 1 for the player with the highest MVP vote share each season, else 0."""
    df = df.copy()
    df["mvp"] = (df["Share"] == df.groupby("Year")["Share"].transform("max")).astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add each player's within-season rank in PPG and team win percentage."""
    df = df.copy()
    df["ppg_rank"] = df.groupby("Year")["PTS"].rank(ascending=False, method="min")
    df["win_rank"] = df.groupby("Year")["W/L%"].rank(ascending=False, method="min")
    return df


def split_by_time(df: pd.DataFrame):
    """Past -> future: train on 1991-2012, test on 2013-2021 (no shuffling)."""
    train = df[df["Year"] <= TRAIN_END]
    test = df[df["Year"] >= TEST_START].copy()
    return train, test
