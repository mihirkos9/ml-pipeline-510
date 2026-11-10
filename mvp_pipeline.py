"""
Simple ML pipeline: predict the NBA MVP from points per game (PPG) and team success.

Data   : NBA player stats 1991-2021 with MVP voting (Dataquest, scraped from Basketball-Reference)
Label  : mvp = 1 if the player had the highest MVP vote share that season, else 0
Features: how each player RANKS within their season in
          (1) points per game and (2) team win percentage
          (an MVP is whoever stands out compared with that year's league)
Split  : train on 1991-2012, test on 2013-2021 (past -> future)
Model  : Logistic Regression
Metric : season-level accuracy = in what share of test seasons is the model's #1 pick the real MVP?
         (row-level accuracy is useless here: ~99.8% of rows are "not MVP")
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

URL = ("https://raw.githubusercontent.com/dataquestio/project-walkthroughs/"
       "571e6565edb7330c65a1dcb35033dac5856ffd6d/mvp/player_mvp_stats.csv")

# 1. Load data and build the label
df = pd.read_csv(URL)
df["mvp"] = (df["Share"] == df.groupby("Year")["Share"].transform("max")).astype(int)
print(f"{len(df):,} player-seasons, {df['Year'].nunique()} seasons, {df['mvp'].sum()} MVPs")

# 2. Features: rank within each season (1 = best)
df["ppg_rank"] = df.groupby("Year")["PTS"].rank(ascending=False, method="min")
df["win_rank"] = df.groupby("Year")["W/L%"].rank(ascending=False, method="min")
FEATURES = ["ppg_rank", "win_rank"]

# 3. Split by time
train = df[df["Year"] <= 2012]
test = df[df["Year"] >= 2013].copy()

# 4. Train a simple model
model = make_pipeline(StandardScaler(),
                      LogisticRegression(class_weight="balanced", max_iter=1000))
model.fit(train[FEATURES], train["mvp"])
test["score"] = model.predict_proba(test[FEATURES])[:, 1]


# 5. Evaluate: pick the top-scoring player each season
def season_accuracy(data, score_col, higher_is_better=True):
    hits = 0
    seasons = data["Year"].unique()
    for year in seasons:
        season = data[data["Year"] == year]
        best = season[score_col].idxmax() if higher_is_better else season[score_col].idxmin()
        hits += season.loc[best, "mvp"] == 1
    return hits / len(seasons), hits, len(seasons)


acc, hits, n = season_accuracy(test, "score")
base_acc, base_hits, _ = season_accuracy(test, "ppg_rank", higher_is_better=False)  # baseline: highest PPG
print(f"\nModel (logistic regression) test accuracy = {acc:.1%} ({hits}/{n} seasons)")
print(f"Baseline (highest PPG)      test accuracy = {base_acc:.1%} ({base_hits}/{n} seasons)")

print("\nSeason-by-season picks:")
for year in sorted(test["Year"].unique()):
    s = test[test["Year"] == year]
    pick = s.loc[s["score"].idxmax(), "Player"]
    real = s.loc[s["mvp"] == 1, "Player"].iloc[0]
    print(f"  {year}: predicted {pick:<24} actual {real:<24} {'OK' if pick == real else 'MISS'}")
