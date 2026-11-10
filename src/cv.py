"""Leave-one-season-out cross-validation on the TRAINING seasons.

This is how the two rank features were chosen (instead of raw stats), using only
the 22 training seasons. The test seasons (2013-2021) are never touched here.

For each training season: hold it out, fit on the other 21, score every player in
the held-out season, and check whether the top-scored player is that season's MVP.
Metric = number of seasons where that pick is correct (same metric as the test set).
"""
import pandas as pd

from src.evaluate import season_picks
from src.features import FEATURE_SETS
from src.model import build_model


def leave_one_season_out(train: pd.DataFrame, features: list):
    """Return (hits, n_seasons) for one feature set."""
    seasons = sorted(train["Year"].unique())
    hits = 0
    for year in seasons:
        fit = train[train["Year"] != year]
        held_out = train[train["Year"] == year]
        model = build_model().fit(fit[features], fit["mvp"])
        scores = pd.Series(model.predict_proba(held_out[features])[:, 1],
                           index=held_out.index)
        hits += int(season_picks(held_out, scores)["correct"].iloc[0])
    return hits, len(seasons)


def compare_feature_sets(train: pd.DataFrame) -> dict:
    """{key: {"label", "features", "hits", "n_seasons", "accuracy"}} for each candidate set."""
    results = {}
    for key, spec in FEATURE_SETS.items():
        hits, n = leave_one_season_out(train, spec["features"])
        results[key] = {"label": spec["label"], "features": spec["features"],
                        "hits": hits, "n_seasons": n, "accuracy": hits / n}
    return results
