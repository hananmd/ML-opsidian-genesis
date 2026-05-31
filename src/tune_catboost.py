"""Optuna HPO for CatBoost, then full CV + submission.

Run:
    python src/tune_catboost.py

Outputs submissions/tuned_catboost.csv
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import catboost as cb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "submissions"
OUT.mkdir(exist_ok=True)

TARGET = "flood_risk_score"
ID = "record_id"
SEED = 42
N_FOLDS = 5
N_TRIALS = 60
GPU_DEVICES = os.getenv("CATBOOST_GPU_DEVICES", "0")
CATBOOST_BASE_PARAMS = dict(
    task_type="GPU",
    devices=GPU_DEVICES,
)

# ── Load & engineer features (same as improved_model.py) ─────────────────────
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

for df in (train, test):
    df["nearest_evac_km"] = pd.to_numeric(df["nearest_evac_km"], errors="coerce")

def engineer_features(train_df, test_df):
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    n_train = len(train_df)

    all_df["rain_x_flood_count"] = all_df["rainfall_7d_mm"].fillna(0) * all_df["historical_flood_count"].fillna(0)
    all_df["elevation_x_river"] = all_df["elevation_m"].fillna(0) * (1 / (all_df["distance_to_river_m"].fillna(1000) + 1))
    all_df["rain_ratio"] = all_df["rainfall_7d_mm"].fillna(0) / (all_df["monthly_rainfall_mm"].fillna(1) + 1)
    all_df["risk_density"] = all_df["historical_flood_count"].fillna(0) * all_df["population_density_per_km2"].fillna(0)
    all_df["infra_x_evac"] = all_df["infrastructure_score"].fillna(50) / (all_df["nearest_evac_km"].fillna(10) + 1)
    all_df["ndwi_x_rain"] = all_df["ndwi"].fillna(0) * all_df["rainfall_7d_mm"].fillna(0)
    all_df["low_elevation_flag"] = (all_df["elevation_m"].fillna(100) < 10).astype(int)
    all_df["close_river_flag"] = (all_df["distance_to_river_m"].fillna(9999) < 500).astype(int)

    coords = all_df[["latitude", "longitude"]].fillna(all_df[["latitude", "longitude"]].median())
    km = KMeans(n_clusters=25, random_state=SEED, n_init=10)
    all_df["geo_cluster"] = km.fit_predict(coords).astype(str)

    te_cols = ["district", "soil_type", "landcover", "urban_rural", "geo_cluster"]
    train_part = all_df.iloc[:n_train].copy()
    train_part[TARGET] = train_df[TARGET].values
    for col in te_cols:
        if col not in all_df.columns:
            continue
        mean_map = train_part.groupby(col)[TARGET].mean()
        global_mean = train_part[TARGET].mean()
        all_df[f"{col}_te"] = all_df[col].map(mean_map).fillna(global_mean)

    return all_df.iloc[:n_train].copy(), all_df.iloc[n_train:].copy()

train, test = engineer_features(train, test)

DROP_COLS = [
    ID, TARGET, "generation_date", "place_name", "reason_not_good_to_live",
    "is_good_to_live", "is_synthetic", "flood_occurrence_current_event",
    "inundation_area_sqm",
]
features = [c for c in train.columns if c not in DROP_COLS and c in test.columns]
cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]

X = train[features].copy()
y = train[TARGET].astype(float)
X_test = test[features].copy()

# Label-encode categoricals for CatBoost (it needs integer cat indices)
le_map = {}
X_enc = X.copy()
X_test_enc = X_test.copy()
for c in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([X[c].astype(str), X_test[c].astype(str)])
    le.fit(combined)
    X_enc[c] = le.transform(X[c].astype(str))
    X_test_enc[c] = le.transform(X_test[c].astype(str))
    le_map[c] = le

cat_indices = [X_enc.columns.tolist().index(c) for c in cat_cols]

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# ── Optuna objective — 3-fold CV on a subsample for speed ────────────────────
kf_tune = KFold(n_splits=3, shuffle=True, random_state=SEED)

def objective(trial):
    params = dict(
        **CATBOOST_BASE_PARAMS,
        loss_function="RMSE",
        random_seed=SEED,
        verbose=0,
        iterations=2000,
        early_stopping_rounds=50,
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 20, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        random_strength=trial.suggest_float("random_strength", 1e-3, 10, log=True),
        border_count=trial.suggest_int("border_count", 32, 255),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 1, 50),
    )
    scores = []
    for tr, va in kf_tune.split(X_enc):
        model = cb.CatBoostRegressor(**params, cat_features=cat_indices)
        model.fit(X_enc.iloc[tr], y.iloc[tr],
                  eval_set=(X_enc.iloc[va], y.iloc[va]),
                  use_best_model=True)
        preds = model.predict(X_enc.iloc[va])
        scores.append(np.sqrt(mean_squared_error(y.iloc[va], preds)))
    return np.mean(scores)

print(f"CatBoost running on GPU device(s): {GPU_DEVICES}")
print(f"Running Optuna HPO — {N_TRIALS} trials (3-fold CV each)...")
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best_params = study.best_params
best_params.update(dict(
    **CATBOOST_BASE_PARAMS,
    loss_function="RMSE",
    random_seed=SEED,
    verbose=0,
    iterations=3000,
    early_stopping_rounds=100,
))
print(f"\nBest trial RMSE (3-fold): {study.best_value:.5f}")
print(f"Best params: {best_params}")

# ── Full 5-fold CV with best params ──────────────────────────────────────────
print(f"\nFull 5-fold CV with tuned params...")
oof = np.zeros(len(X_enc))
preds_test = np.zeros(len(X_test_enc))

for fold, (tr, va) in enumerate(kf.split(X_enc), 1):
    model = cb.CatBoostRegressor(**best_params, cat_features=cat_indices)
    model.fit(X_enc.iloc[tr], y.iloc[tr],
              eval_set=(X_enc.iloc[va], y.iloc[va]),
              use_best_model=True)
    oof[va] = model.predict(X_enc.iloc[va])
    preds_test += model.predict(X_test_enc) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(y.iloc[va], oof[va]))
    print(f"  fold {fold}: rmse={rmse:.5f}  iter={model.best_iteration_}")

oof_rmse = np.sqrt(mean_squared_error(y, oof))
print(f"\nTuned CatBoost OOF RMSE: {oof_rmse:.5f}")
print(f"Previous best (ensemble): 0.23511")
print(f"Improvement: {0.23511 - oof_rmse:+.5f}")

final_preds = np.clip(preds_test, 0, 1)
sub = pd.DataFrame({ID: test[ID], TARGET: final_preds})
sub.to_csv(OUT / "tuned_catboost.csv", index=False)
print(f"\nWrote {OUT / 'tuned_catboost.csv'}  (NOT uploaded)")
