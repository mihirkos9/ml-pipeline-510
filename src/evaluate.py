"""Season-level evaluation: in how many test seasons is our #1 pick the real MVP?

Row-level accuracy is useless here (~99.8% of rows are "not MVP"), so each
method (the model and every baseline) is scored the same way: in each season,
pick the player with the highest score and check whether they won MVP.

To add a baseline, add one entry to BASELINES. It then shows up automatically in
the printout, metrics.json, and test_predictions.csv.
"""
import pandas as pd


def eff_per_game(d: pd.DataFrame) -> pd.Series:
    """NBA 'EFF' efficiency per game: a simple, transparent stand-in for PER.

    EFF = PTS + REB + AST + STL + BLK - missed FG - missed FT - TOV
    (Real PER needs league pace and constants that aren't in this dataset.)
    """
    return (d["PTS"] + d["TRB"] + d["AST"] + d["STL"] + d["BLK"]
            - (d["FGA"] - d["FG"]) - (d["FTA"] - d["FT"]) - d["TOV"])


# key -> {"label": printed name, "score": function(test_df) -> Series (higher = better)}
BASELINES = {
    "highest_ppg": {
        "label": "Baseline (highest PPG)",
        "score": lambda d: d["PTS"],
    },
    "highest_eff": {
        "label": "Baseline (highest EFF)",
        "score": eff_per_game,
    },
}


def season_picks(data: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    """For each season, the top-scoring player, the real MVP, and whether they match."""
    d = data.assign(_score=scores)
    rows = []
    for year, season in d.groupby("Year"):
        best = season["_score"].idxmax()
        rows.append({
            "Year": int(year),
            "pick": season.loc[best, "Player"],
            "actual": season.loc[season["mvp"] == 1, "Player"].iloc[0],
            "correct": bool(season.loc[best, "mvp"] == 1),
        })
    return pd.DataFrame(rows)


def evaluate_all(test: pd.DataFrame, model_scores: pd.Series):
    """Score the model and every baseline on the test seasons.

    Returns (results, predictions):
      results     : {key: {"label", "accuracy", "hits", "n_seasons"}}
      predictions : one row per test season with each method's pick and correctness
    """
    methods = {"model": ("Model (logistic regression)", model_scores)}
    for key, b in BASELINES.items():
        methods[key] = (b["label"], b["score"](test))

    results, predictions = {}, None
    for key, (label, scores) in methods.items():
        picks = season_picks(test, scores)
        results[key] = {
            "label": label,
            "accuracy": float(picks["correct"].mean()),
            "hits": int(picks["correct"].sum()),
            "n_seasons": int(len(picks)),
        }
        wide = picks.rename(columns={"pick": f"{key}_pick", "correct": f"{key}_correct"})
        if predictions is None:
            predictions = wide.rename(columns={"actual": "actual_mvp"})
        else:
            predictions = predictions.merge(
                wide.drop(columns="actual"), on="Year")
    return results, predictions
