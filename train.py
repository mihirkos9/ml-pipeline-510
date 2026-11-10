"""Simple ML pipeline: predict the NBA MVP from points per game (PPG) and team success.

Data    : NBA player stats 1991-2021 with MVP voting (src/data.py)
Label   : mvp = 1 if the player had the highest MVP vote share that season (src/features.py)
Features: each player's within-season RANK in PPG and team win percentage
Split   : train on 1991-2012, test on 2013-2021 (past -> future)
Model   : Logistic Regression (src/model.py)
Metric  : season-level accuracy (src/evaluate.py)

Usage: python train.py
"""
import pandas as pd

from src.data import load_data
from src.evaluate import evaluate_all
from src.features import FEATURES, add_features, add_label, split_by_time
from src.model import build_model


def main():
    # 1. Load data, build label and features
    df = add_features(add_label(load_data()))
    print(f"{len(df):,} player-seasons, {df['Year'].nunique()} seasons, {df['mvp'].sum()} MVPs")

    # 2. Split by time
    train, test = split_by_time(df)

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


if __name__ == "__main__":
    main()
