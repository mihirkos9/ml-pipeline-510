"""Simple ML pipeline: predict the NBA MVP from points per game (PPG) and team success.

Data    : NBA player stats 1991-2021 with MVP voting (src/data.py)
Label   : mvp = 1 if the player had the highest MVP vote share that season (src/features.py)
Features: each player's within-season RANK in PPG and team win percentage
Split   : train on 1991-2012, test on 2013-2021 (past -> future)
Model   : Logistic Regression (src/model.py)
Metric  : season-level accuracy (src/evaluate.py)
CV      : leave-one-season-out on the training seasons picks the features (src/cv.py)

Usage: python train.py
"""
import json
from pathlib import Path

import pandas as pd

from src.cv import compare_feature_sets
from src.data import load_data
from src.evaluate import evaluate_all
from src.features import FEATURES, add_features, add_label, split_by_time
from src.model import build_model

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def save_results(results, predictions, cv_results, train, test):
    """Write results/metrics.json and results/test_predictions.csv (all methods, incl. baselines)."""
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics = {
        "train_seasons": f"{int(train['Year'].min())}-{int(train['Year'].max())}",
        "test_seasons": f"{int(test['Year'].min())}-{int(test['Year'].max())}",
        "test": {k: {**r, "accuracy": round(r["accuracy"], 4)} for k, r in results.items()},
        "cv_leave_one_season_out": {
            k: {"label": r["label"], "features": r["features"], "hits": r["hits"],
                "n_seasons": r["n_seasons"], "accuracy": round(r["accuracy"], 4)}
            for k, r in cv_results.items()},
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    first = ["Year", "actual_mvp"]
    predictions[first + [c for c in predictions.columns if c not in first]].to_csv(
        RESULTS_DIR / "test_predictions.csv", index=False)


def main():
    # 1. Load data, build label and features
    df = add_features(add_label(load_data()))
    print(f"{len(df):,} player-seasons, {df['Year'].nunique()} seasons, {df['mvp'].sum()} MVPs")

    # 2. Split by time
    train, test = split_by_time(df)

    # 2b. Feature selection evidence: leave-one-season-out CV on the TRAINING seasons only
    cv_results = compare_feature_sets(train)
    print(f"\nLeave-one-season-out CV on the {train['Year'].nunique()} training seasons "
          "(how the features were chosen):")
    for r in cv_results.values():
        print(f"  {r['label']:<46} {r['hits']}/{r['n_seasons']} seasons ({r['accuracy']:.1%})")

    # 3. Train a simple model
    model = build_model()
    model.fit(train[FEATURES], train["mvp"])
    model_scores = pd.Series(model.predict_proba(test[FEATURES])[:, 1], index=test.index)

    # 4. Evaluate the model and the baselines on the test seasons
    results, predictions = evaluate_all(test, model_scores)
    print()
    for r in results.values():
        print(f"{r['label']:<27} test accuracy = {r['accuracy']:.1%} "
              f"({r['hits']}/{r['n_seasons']} seasons)")

    print("\nSeason-by-season picks:")
    for row in predictions.itertuples():
        status = "OK" if row.model_correct else "MISS"
        print(f"  {row.Year}: predicted {row.model_pick:<24} actual {row.actual_mvp:<24} {status}")

    # 5. Save results for the README and for review
    save_results(results, predictions, cv_results, train, test)
    print(f"\nSaved results to {RESULTS_DIR.name}/metrics.json and "
          f"{RESULTS_DIR.name}/test_predictions.csv")


if __name__ == "__main__":
    main()
