"""Improved ensemble model: LightGBM + XGBoost + CatBoost with feature engineering.

Run:
    python src/improved_model.py

Outputs OOF RMSE and writes submissions/improved_ensemble.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "submissions"
OUT.mkdir(exist_ok=True)

TARGET = "flood_risk_score"
ID = "record_id"
SEED = 42
N_FOLDS = 5

# ── Load data ────────────────────────────────────────────────────────────────
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

for df in (train, test):
    df["nearest_evac_km"] = pd.to_numeric(df["nearest_evac_km"], errors="coerce")

# ── Feature Engineering ──────────────────────────────────────────────────────
def engineer_features(train_df, test_df):
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    n_train = len(train_df)

    # Interaction features
    all_df["rain_x_flood_count"] = (
        all_df["rainfall_7d_mm"].fillna(0) * all_df["historical_flood_count"].fillna(0)
    )
    all_df["elevation_x_river"] = (
        all_df["elevation_m"].fillna(0) * (1 / (all_df["distance_to_river_m"].fillna(1000) + 1))
    )
    all_df["rain_ratio"] = (
        all_df["rainfall_7d_mm"].fillna(0) / (all_df["monthly_rainfall_mm"].fillna(1) + 1)
    )
    all_df["risk_density"] = (
        all_df["historical_flood_count"].fillna(0) * all_df["population_density_per_km2"].fillna(0)
    )
    all_df["infra_x_evac"] = (
        all_df["infrastructure_score"].fillna(50) / (all_df["nearest_evac_km"].fillna(10) + 1)
    )
    all_df["ndwi_x_rain"] = (
        all_df["ndwi"].fillna(0) * all_df["rainfall_7d_mm"].fillna(0)
    )
    all_df["low_elevation_flag"] = (all_df["elevation_m"].fillna(100) < 10).astype(int)
    all_df["close_river_flag"] = (all_df["distance_to_river_m"].fillna(9999) < 500).astype(int)

    # Geospatial clustering
    coords = all_df[["latitude", "longitude"]].fillna(all_df[["latitude", "longitude"]].median())
    km = KMeans(n_clusters=25, random_state=SEED, n_init=10)
    all_df["geo_cluster"] = km.fit_predict(coords).astype(str)

    # Target encoding (fit only on train portion, apply to all)
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

# ── Feature selection ────────────────────────────────────────────────────────
DROP_COLS = [
    ID, TARGET, "generation_date", "place_name", "reason_not_good_to_live",
    "is_good_to_live", "is_synthetic", "flood_occurrence_current_event",
    "inundation_area_sqm",
]
features = [c for c in train.columns if c not in DROP_COLS and c in test.columns]

cat_cols_lgb = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]
print(f"Features: {len(features)}  |  Categoricals: {len(cat_cols_lgb)}")

X = train[features].copy()
y = train[TARGET].astype(float)
X_test = test[features].copy()

# Encode categoricals for XGBoost/CatBoost
X_enc = X.copy()
X_test_enc = X_test.copy()
for c in cat_cols_lgb:
    le = LabelEncoder()
    combined = pd.concat([X[c].astype(str), X_test[c].astype(str)])
    le.fit(combined)
    X_enc[c] = le.transform(X[c].astype(str))
    X_test_enc[c] = le.transform(X_test[c].astype(str))

# ── Cross-validation ─────────────────────────────────────────────────────────
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
preds_lgb = np.zeros(len(X_test))
preds_xgb = np.zeros(len(X_test))
preds_cat = np.zeros(len(X_test))

lgb_params = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=127,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_data_in_leaf=30,
    lambda_l1=0.1,
    lambda_l2=0.1,
    verbose=-1,
    seed=SEED,
)

xgb_params = dict(
    objective="reg:squarederror",
    eval_metric="rmse",
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_estimators=3000,
    early_stopping_rounds=100,
    verbosity=0,
    random_state=SEED,
)

cat_params = dict(
    loss_function="RMSE",
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    iterations=3000,
    early_stopping_rounds=100,
    random_seed=SEED,
    verbose=0,
)

print("\n" + "="*60)
print("Training LightGBM")
print("="*60)
for fold, (tr, va) in enumerate(kf.split(X), 1):
    Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
    ytr, yva = y.iloc[tr], y.iloc[va]
    for c in cat_cols_lgb:
        Xtr[c] = Xtr[c].astype("category")
        Xva[c] = pd.Categorical(Xva[c], categories=Xtr[c].cat.categories)
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat_cols_lgb)
    dva = lgb.Dataset(Xva, yva, categorical_feature=cat_cols_lgb)
    model = lgb.train(lgb_params, dtr, num_boost_round=3000,
                      valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    Xt = X_test.copy()
    for c in cat_cols_lgb:
        Xt[c] = pd.Categorical(Xt[c], categories=Xtr[c].cat.categories)
    oof_lgb[va] = model.predict(Xva, num_iteration=model.best_iteration)
    preds_lgb += model.predict(Xt, num_iteration=model.best_iteration) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(yva, oof_lgb[va]))
    print(f"  fold {fold}: rmse={rmse:.5f}  iter={model.best_iteration}")

lgb_oof_rmse = np.sqrt(mean_squared_error(y, oof_lgb))
print(f"LightGBM OOF RMSE: {lgb_oof_rmse:.5f}")

print("\n" + "="*60)
print("Training XGBoost")
print("="*60)
for fold, (tr, va) in enumerate(kf.split(X_enc), 1):
    Xtr, Xva = X_enc.iloc[tr], X_enc.iloc[va]
    ytr, yva = y.iloc[tr], y.iloc[va]
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    oof_xgb[va] = model.predict(Xva)
    preds_xgb += model.predict(X_test_enc) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(yva, oof_xgb[va]))
    print(f"  fold {fold}: rmse={rmse:.5f}  iter={model.best_iteration}")

xgb_oof_rmse = np.sqrt(mean_squared_error(y, oof_xgb))
print(f"XGBoost OOF RMSE: {xgb_oof_rmse:.5f}")

print("\n" + "="*60)
print("Training CatBoost")
print("="*60)
cat_feature_indices = [X_enc.columns.tolist().index(c) for c in cat_cols_lgb if c in X_enc.columns]
for fold, (tr, va) in enumerate(kf.split(X_enc), 1):
    Xtr, Xva = X_enc.iloc[tr], X_enc.iloc[va]
    ytr, yva = y.iloc[tr], y.iloc[va]
    model = cb.CatBoostRegressor(**cat_params, cat_features=cat_feature_indices)
    model.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
    oof_cat[va] = model.predict(Xva)
    preds_cat += model.predict(X_test_enc) / N_FOLDS
    rmse = np.sqrt(mean_squared_error(yva, oof_cat[va]))
    print(f"  fold {fold}: rmse={rmse:.5f}  iter={model.best_iteration_}")

cat_oof_rmse = np.sqrt(mean_squared_error(y, oof_cat))
print(f"CatBoost OOF RMSE: {cat_oof_rmse:.5f}")

# ── Optimal ensemble weights ─────────────────────────────────────────────────
from scipy.optimize import minimize

def blend_rmse(w):
    w = np.array(w)
    w = w / w.sum()
    blended = w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat
    return np.sqrt(mean_squared_error(y, blended))

result = minimize(blend_rmse, x0=[1/3, 1/3, 1/3],
                  method="Nelder-Mead",
                  options={"maxiter": 1000, "xatol": 1e-6})
best_w = np.array(result.x)
best_w = best_w / best_w.sum()

print(f"\n{'='*60}")
print(f"Optimal blend weights:")
print(f"  LightGBM: {best_w[0]:.3f}")
print(f"  XGBoost:  {best_w[1]:.3f}")
print(f"  CatBoost: {best_w[2]:.3f}")

# Simple average as fallback
simple_avg_rmse = np.sqrt(mean_squared_error(y, (oof_lgb + oof_xgb + oof_cat) / 3))
optimal_rmse = blend_rmse(best_w)
print(f"\nSimple average OOF RMSE:  {simple_avg_rmse:.5f}")
print(f"Optimal blend OOF RMSE:   {optimal_rmse:.5f}")
print(f"Baseline LightGBM RMSE:   0.23585")
print(f"Improvement over baseline: {0.23585 - optimal_rmse:.5f}")

# Use optimal weights for final predictions
final_preds = best_w[0]*preds_lgb + best_w[1]*preds_xgb + best_w[2]*preds_cat
final_preds = np.clip(final_preds, 0, 1)

sub = pd.DataFrame({ID: test[ID], TARGET: final_preds})
sub.to_csv(OUT / "improved_ensemble.csv", index=False)
print(f"\nWrote {OUT / 'improved_ensemble.csv'}  (NOT uploaded)")
