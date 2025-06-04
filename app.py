import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

# Load and preprocess data
matches = pd.read_csv('matches.csv', index_col=0)
matches["date"] = pd.to_datetime(matches["date"])
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype(int)
matches["day_code"] = matches["date"].dt.dayofweek
matches["target"] = (matches["result"] == "W").astype(int)

# Rolling average feature engineering
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    return group.dropna(subset=new_cols)

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

matches_rolling = matches.groupby("team").apply(lambda g: rolling_averages(g, cols, new_cols))
matches_rolling = matches_rolling.droplevel("team").reset_index(drop=True)

# Model and prediction function
predictors = ["venue_code", "opp_code", "hour", "day_code"] + new_cols
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

def make_predictions(data, predictors): 
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] >= '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
    precision = precision_score(test["target"], preds)
    return combined, precision

combined, precision = make_predictions(matches_rolling, predictors)

# Merge predictions with match metadata
combined = combined.merge(
    matches_rolling[["date", "team", "opponent", "result"]],
    left_index=True,
    right_index=True
)

# A custom dictionary to handle missing keys
class MissingDict(dict):
    __missing__ = lambda self, key: key
    
# Mapping team names to their common names
mapping = MissingDict(**{
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle",
    "Tottenham Hotspur": "Tottenham",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham"
})

combined["new_team"] = combined["team"].map(mapping)

# Merge team predictions to simulate head-to-head outcomes
merged = combined.merge(
    combined,
    left_on=["date", "new_team"],
    right_on=["date", "opponent"],
    suffixes=("_x", "_y")
)

# Example analysis: predicted wins and actual results
results = merged[
    (merged["prediction_x"] == 1) & (merged["prediction_y"] == 0)
]["actual_x"].value_counts()

print("Precision:", precision)


