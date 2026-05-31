"""Baseline LightGBM regressor for flood_risk_score.

Run:
    python src/baseline_lgbm.py

Outputs OOF RMSE and writes submissions/baseline_lgbm.csv (does NOT upload).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "submissions"
OUT.mkdir(exist_ok=True)

TARGET = "flood_risk_score"
ID = "record_id"
DROP = [ID, TARGET, "generation_date", "place_name", "reason_not_good_to_live"]

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

# test.csv has nearest_evac_km as object (stray strings) in some rows — coerce
for df in (train, test):
    df["nearest_evac_km"] = pd.to_numeric(df["nearest_evac_km"], errors="coerce")

features = [c for c in train.columns if c not in DROP and c in test.columns]
cat_cols = [
    c for c in features
    if not pd.api.types.is_numeric_dtype(train[c])
]

for c in cat_cols:
    train[c] = train[c].astype("category")
    test[c] = pd.Categorical(test[c], categories=train[c].cat.categories)

X, y = train[features], train[TARGET].astype(float)
X_test = test[features]

params = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.05,
    num_leaves=63,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    min_data_in_leaf=50,
    verbose=-1,
    seed=42,
)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
preds = np.zeros(len(X_test))

for fold, (tr, va) in enumerate(kf.split(X), 1):
    dtr = lgb.Dataset(X.iloc[tr], y.iloc[tr], categorical_feature=cat_cols)
    dva = lgb.Dataset(X.iloc[va], y.iloc[va], categorical_feature=cat_cols)
    model = lgb.train(
        params, dtr, num_boost_round=3000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    oof[va] = model.predict(X.iloc[va], num_iteration=model.best_iteration)
    preds += model.predict(X_test, num_iteration=model.best_iteration) / kf.n_splits
    print(f"fold {fold}: rmse={np.sqrt(mean_squared_error(y.iloc[va], oof[va])):.5f}  best_iter={model.best_iteration}")

rmse = float(np.sqrt(mean_squared_error(y, oof)))
print(f"\nOOF RMSE: {rmse:.5f}")

sub = pd.DataFrame({ID: test[ID], TARGET: np.clip(preds, 0, 1)})
sub.to_csv(OUT / "baseline_lgbm.csv", index=False)
print(f"Wrote {OUT / 'baseline_lgbm.csv'}  (NOT uploaded)")
