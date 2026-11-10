"""Fast, offline checks on the pipeline's building blocks (no download needed).

Run from the repo root:  python -m pytest
"""
import pandas as pd

from src.evaluate import season_picks
from src.features import (FEATURES, TEST_START, TRAIN_END, add_features,
                          add_label, split_by_time)

# Columns that come from MVP voting itself. A model that sees them is cheating.
LEAKY_COLUMNS = {"Share", "Pts Won", "Pts Max", "mvp"}


def toy_data() -> pd.DataFrame:
    """Two players in each of four seasons: two training seasons, two test seasons."""
    rows = []
    for year in [TRAIN_END - 1, TRAIN_END, TEST_START, TEST_START + 1]:
        rows.append({"Player": f"Star {year}", "Year": year, "PTS": 30.0, "W/L%": 0.75, "Share": 0.9})
        rows.append({"Player": f"Role {year}", "Year": year, "PTS": 10.0, "W/L%": 0.40, "Share": 0.0})
    return pd.DataFrame(rows)


def test_features_do_not_use_mvp_voting_columns():
    assert LEAKY_COLUMNS.isdisjoint(FEATURES)


def test_exactly_one_mvp_per_season():
    df = add_label(toy_data())
    assert (df.groupby("Year")["mvp"].sum() == 1).all()


def test_rank_features_rank_within_each_season():
    df = add_features(toy_data())
    stars = df[df["Player"].str.startswith("Star")]
    assert (stars["ppg_rank"] == 1).all() and (stars["win_rank"] == 1).all()


def test_train_and_test_seasons_do_not_overlap():
    train, test = split_by_time(toy_data())
    assert set(train["Year"]).isdisjoint(test["Year"])
    assert train["Year"].max() <= TRAIN_END
    assert test["Year"].min() >= TEST_START


def test_season_picks_takes_top_score_each_season():
    df = add_label(toy_data())
    # Score the role player highest in the first season only: that season should be a miss.
    scores = df["PTS"].copy()
    first_role = df.index[(df["Year"] == df["Year"].min()) & df["Player"].str.startswith("Role")][0]
    scores[first_role] = 100.0
    picks = season_picks(df, scores)
    assert len(picks) == 4
    assert picks["correct"].tolist() == [False, True, True, True]
