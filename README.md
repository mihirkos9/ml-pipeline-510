# NBA MVP Predictor (Simple ML Pipeline)

Can we predict the NBA MVP using points per game (PPG) and team success?

## Results

| Method | Test accuracy |
|---|---|
| **Logistic Regression** | **test accuracy = 77.8%** (picked the real MVP in 7 of 9 seasons) |
| Baseline: player with the highest PPG | test accuracy = 44.4% (4 of 9 seasons) |

**Accuracy here means season-level accuracy:** each test season, the model scores every player and we pick the top one. It counts as correct if that player is the real MVP.
Normal row-level accuracy would be meaningless: ~99.8% of player-seasons are "not MVP," so predicting "no" every time scores ~99.8%.

### Test-season picks (2013-2021)

| Season | Predicted | Actual MVP | Correct? |
|---|---|---|---|
| 2013 | LeBron James | LeBron James | Yes |
| 2014 | Kevin Durant | Kevin Durant | Yes |
| 2015 | Stephen Curry | Stephen Curry | Yes |
| 2016 | Stephen Curry | Stephen Curry | Yes |
| 2017 | Stephen Curry | Russell Westbrook | No |
| 2018 | James Harden | James Harden | Yes |
| 2019 | Giannis Antetokounmpo | Giannis Antetokounmpo | Yes |
| 2020 | Giannis Antetokounmpo | Giannis Antetokounmpo | Yes |
| 2021 | Donovan Mitchell | Nikola Jokic | No |

## Pipeline

1. **Data:** NBA player stats with MVP voting, 1991-2021 (14,092 player-seasons), from the [Dataquest NBA MVP project](https://github.com/dataquestio/project-walkthroughs/tree/master/mvp), originally scraped from Basketball-Reference. The script downloads it directly from GitHub (pinned to a fixed commit).
2. **Label:** `mvp = 1` for the player with the highest MVP vote share each season, otherwise 0.
3. **Features (2):** each player's **rank within their season** in points per game and in team win percentage (1 = best). Ranks work better than raw stats because an MVP is whoever stands out compared with that year's league, and scoring levels change over eras.
4. **Split by time:** train on 1991-2012 (22 seasons), test on 2013-2021 (9 seasons), so the model predicts the "future" and never sees test seasons.
5. **Model:** Logistic Regression (with standardized features and balanced class weights because MVPs are rare).
6. **Evaluation:** season-level accuracy on the test seasons, compared with a "highest PPG" baseline.

## How the features were chosen

I compared a few simple feature sets using only the 22 training seasons (leave-one-season-out: hold out one season, train on the rest, check whether the model picks that season's MVP). Rank features for PPG + team win % did best at about 68% (15 of 22 seasons), ahead of raw stats like assists, rebounds, steals, and blocks (13 of 22). The test seasons were used only once, for the final result above.

## Takeaways

- Scoring alone is not enough: the top scorer was the MVP only 44.4% of the time. Adding team success raised that to 77.8%.
- With only 9 test seasons, each season is worth ~11 percentage points, so the result is noisy.
- Misses: 2017 (Westbrook's triple-double season beat Curry and Harden on narrative, which the model can't see) and 2021 (Jokic won on all-around play; the model has no passing or rebounding features, and it picked Donovan Mitchell).
- Ideas to improve: add an efficiency stat, a minimum-games filter, or more seasons of data.

## Run it

```bash
pip install -r requirements.txt
python mvp_pipeline.py
```
