# NBA MVP Predictor (Simple ML Pipeline)

Can we predict the NBA MVP using points per game (PPG) and team success?

## Results

| Method | Test accuracy |
|---|---|
| **Logistic Regression (our model)** | **test accuracy = 77.8%** (picked the real MVP in 7 of 9 seasons) |
| Baseline: highest EFF (PER-style efficiency) | test accuracy = 77.8% (7 of 9 seasons) |
| Baseline: highest PPG | test accuracy = 44.4% (4 of 9 seasons) |

**Accuracy here means season-level accuracy:** each test season, every method scores every player and we pick the top one. It counts as correct if that player is the real MVP.
Normal row-level accuracy would be meaningless: ~99.8% of player-seasons are "not MVP," so predicting "no" every time scores ~99.8%.

**Honest read:** the model clearly beats the scoring-only baseline, and **ties** a simple efficiency baseline. The two miss *different* seasons (see below), so they capture different things.

### Test-season picks (2013-2021)

| Season | Model pick | EFF baseline pick | Actual MVP | Model correct? |
|---|---|---|---|---|
| 2013 | LeBron James | LeBron James | LeBron James | Yes |
| 2014 | Kevin Durant | Kevin Durant | Kevin Durant | Yes |
| 2015 | Stephen Curry | Anthony Davis | Stephen Curry | Yes |
| 2016 | Stephen Curry | Stephen Curry | Stephen Curry | Yes |
| 2017 | Stephen Curry | Russell Westbrook | Russell Westbrook | No |
| 2018 | James Harden | Anthony Davis | James Harden | Yes |
| 2019 | Giannis Antetokounmpo | Giannis Antetokounmpo | Giannis Antetokounmpo | Yes |
| 2020 | Giannis Antetokounmpo | Giannis Antetokounmpo | Giannis Antetokounmpo | Yes |
| 2021 | Donovan Mitchell | Nikola Jokic | Nikola Jokic | No |

## Pipeline

1. **Data** (`src/data.py`): NBA player stats with MVP voting, 1991-2021 (14,092 player-seasons), from the [Dataquest NBA MVP project](https://github.com/dataquestio/project-walkthroughs/tree/571e6565edb7330c65a1dcb35033dac5856ffd6d/mvp), originally scraped from [Basketball-Reference](https://www.basketball-reference.com/). Loaded from a URL **pinned to commit `571e656`**, so the data can't change underneath us.
2. **Label** (`src/features.py`): `mvp = 1` for the player with the highest MVP vote share each season, otherwise 0.
3. **Features (2)** (`src/features.py`): each player's **rank within their season** in points per game and in team win percentage (1 = best). Ranks work better than raw stats because an MVP is whoever stands out compared with that year's league, and scoring levels change over eras. MVP-voting columns (`Share`, `Pts Won`, `Pts Max`) are never used as features.
4. **Split by time** (`src/features.py`): train on 1991-2012 (22 seasons), test on 2013-2021 (9 seasons). No shuffling: the model predicts the "future" and never sees test seasons.
5. **Model** (`src/model.py`): Logistic Regression with standardized features and balanced class weights (MVPs are ~1 in 450 rows).
6. **Evaluation** (`src/evaluate.py`): season-level accuracy on the test seasons, for the model and two baselines scored the same way.

## How the features were chosen

We compared simple feature sets using **only the 22 training seasons** (leave-one-season-out: hold out one season, train on the rest, check whether the model picks that season's MVP). PPG rank + team win % rank did best at 15 of 22 seasons (~68%), ahead of six raw stats (PPG, assists, rebounds, steals, blocks, win %) at 13 of 22. The test seasons were used only once, for the final result above.

## Reproducibility

- **Pinned data:** the dataset URL points to a fixed Git commit.
- **Pinned environment:** exact package versions in `requirements.txt` (Python 3.11+).
- **Deterministic:** no random shuffling; logistic regression (lbfgs) gives the same result every run.
- **Tests:** `tests/test_pipeline.py` checks for leakage (no MVP-voting columns in the features), that train and test seasons don't overlap, that each season has exactly one MVP, and that season-level scoring picks the top player. The tests use a small made-up table, so they run offline in under a second.

## Takeaways

- Scoring alone is not enough: the top scorer was the MVP only 44.4% of the time. Adding team success raised that to 77.8%.
- The model and the EFF baseline are complementary. EFF misses Curry (2015) and Harden (2018), whose cases relied on team success; the model misses Westbrook (2017) and Jokic (2021), who won on all-around production (rebounds, assists) that two features can't see.
- With only 9 test seasons, each season is worth ~11 percentage points, so these numbers are noisy.
- Ideas to improve: add EFF or rebounds/assists as model features, a games-played feature, or more seasons of data. Any new feature should be chosen with the training-season cross-validation, not by looking at test misses.

## Project structure

```
train.py              # entry point: load -> features -> split -> train -> evaluate
src/data.py           # pinned data download
src/features.py       # label, rank features, time-based split
src/model.py          # model definition
src/evaluate.py       # season-level accuracy + baselines (PPG, EFF)
tests/test_pipeline.py
requirements.txt
```

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py          # prints test accuracy for the model and both baselines
python -m pytest         # 5 offline checks
```
