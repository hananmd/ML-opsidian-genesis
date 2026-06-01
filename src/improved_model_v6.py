"""v6 = v5 + tuned CatBoost params (Optuna) + isotonic regression calibration.

Changes from v5:
  (A) CatBoost params replaced with Optuna best: lr=0.048, depth=5, border_count=40
  (B) Calibration: isotonic regression (OOF fit) instead of global linear scalar.
      IsotonicRegression learns a monotone mapping from OOF preds → targets,
      recovering the non-uniform compression trees produce in the tails.

Run:
    python src/improved_model_v6.py
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize, minimize_scalar
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "submissions"
OUT.mkdir(exist_ok=True)

TARGET = "flood_risk_score"
ID = "record_id"
SEED = 42
N_FOLDS = 5
N_SPATIAL_CLUSTERS = 20


def custom_metric(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return (mae + rmse) / 2 * (1 + max(0, 1 - r2))


# ─── Load ───────────────────────────────────────────────────────────────────
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
for df in (train, test):
    df["nearest_evac_km"] = pd.to_numeric(df["nearest_evac_km"], errors="coerce")
print(f"Train: {train.shape}  Test: {test.shape}")

# ─── Missingness flags ──────────────────────────────────────────────────────
MISSING_FLAG_COLS = [
    "electricity", "road_quality", "distance_to_river_m", "drainage_index",
    "ndvi", "ndwi", "infrastructure_score", "nearest_evac_km",
    "nearest_hospital_km", "soil_type", "population_density_per_km2",
    "built_up_percent",
]
for df in (train, test):
    for c in MISSING_FLAG_COLS:
        if c in df.columns:
            df[f"{c}_missing"] = df[c].isnull().astype(int)

# ─── Interactions ───────────────────────────────────────────────────────────
def add_interactions(df):
    df = df.copy()
    df["rain_x_flood_count"] = df["rainfall_7d_mm"].fillna(0) * df["historical_flood_count"].fillna(0)
    df["elevation_x_river"] = df["elevation_m"].fillna(0) * (1 / (df["distance_to_river_m"].fillna(1000) + 1))
    df["rain_ratio"] = df["rainfall_7d_mm"].fillna(0) / (df["monthly_rainfall_mm"].fillna(1) + 1)
    df["risk_density"] = df["historical_flood_count"].fillna(0) * df["population_density_per_km2"].fillna(0)
    df["infra_x_evac"] = df["infrastructure_score"].fillna(50) / (df["nearest_evac_km"].fillna(10) + 1)
    df["ndwi_x_rain"] = df["ndwi"].fillna(0) * df["rainfall_7d_mm"].fillna(0)
    df["low_elevation_flag"] = (df["elevation_m"].fillna(100) < 10).astype(int)
    df["close_river_flag"] = (df["distance_to_river_m"].fillna(9999) < 500).astype(int)
    return df

train = add_interactions(train)
test = add_interactions(test)

# ─── District stats ─────────────────────────────────────────────────────────
DIST_STAT_COLS = [
    "historical_flood_count", "rainfall_7d_mm", "monthly_rainfall_mm",
    "infrastructure_score", "elevation_m", "distance_to_river_m",
    "drainage_index", "ndwi", "population_density_per_km2",
]
for col in DIST_STAT_COLS:
    if col not in train.columns:
        continue
    grp = train.groupby("district")[col]
    mean_map, std_map = grp.mean(), grp.std()
    gm, gs = train[col].mean(), train[col].std()
    for df in (train, test):
        df[f"district_{col}_mean"] = df["district"].map(mean_map).fillna(gm)
        df[f"district_{col}_std"] = df["district"].map(std_map).fillna(gs)

# ─── Relative features ──────────────────────────────────────────────────────
for raw, ref in [
    ("elevation_m", "district_elevation_m_mean"),
    ("distance_to_river_m", "district_distance_to_river_m_mean"),
    ("rainfall_7d_mm", "district_rainfall_7d_mm_mean"),
    ("infrastructure_score", "district_infrastructure_score_mean"),
    ("drainage_index", "district_drainage_index_mean"),
]:
    if raw in train.columns and ref in train.columns:
        for df in (train, test):
            df[f"rel_{raw}"] = df[raw] - df[ref]

for df in (train, test):
    df["rainfall_district_zscore"] = (
        (df["rainfall_7d_mm"] - df["district_rainfall_7d_mm_mean"])
        / (df["district_rainfall_7d_mm_std"] + 0.01)
    )

# ─── Compound risk ──────────────────────────────────────────────────────────
drain_min, drain_max = train["drainage_index"].min(), train["drainage_index"].max()
drain_median = train["drainage_index"].median()
for df in (train, test):
    drain = df["drainage_index"].fillna(drain_median)
    sat = df["monthly_rainfall_mm"].fillna(0) * (1 - (drain - drain_min) / (drain_max - drain_min + 1e-9))
    df["catchment_saturation_proxy"] = sat
    df["compound_flood_risk"] = (
        (df["rainfall_7d_mm"].fillna(0) * sat) / (df["elevation_m"].fillna(0).clip(lower=0) + 1)
    )
    df["historical_risk_amplifier"] = (df["historical_flood_count"].fillna(0) + 1) * sat
    df["evacuation_risk"] = df["nearest_evac_km"].fillna(10) / (df["infrastructure_score"].fillna(50) + 0.1)
    df["triple_risk"] = (
        df["rainfall_7d_mm"].fillna(0)
        * df["historical_flood_count"].fillna(0)
        * df["low_elevation_flag"]
    )

# ─── DBSCAN isolation (v3 settings) ─────────────────────────────────────────
all_coords = pd.concat([train[["latitude", "longitude"]], test[["latitude", "longitude"]]])
all_coords = all_coords.fillna(all_coords.median())
dbscan = DBSCAN(eps=3.0 / 6371.0, min_samples=20, algorithm="ball_tree", metric="haversine")
labels = dbscan.fit_predict(np.radians(all_coords.values))
train["is_isolated_location"] = (labels[:len(train)] == -1).astype(int)
test["is_isolated_location"] = (labels[len(train):] == -1).astype(int)

coords_train = train[["latitude", "longitude"]].fillna(train[["latitude", "longitude"]].median())
spatial_groups = KMeans(n_clusters=N_SPATIAL_CLUSTERS, random_state=SEED, n_init=10).fit_predict(coords_train)

DROP_COLS = [
    ID, TARGET, "generation_date", "place_name", "reason_not_good_to_live",
    "is_good_to_live", "is_synthetic", "flood_occurrence_current_event",
    "inundation_area_sqm",
    "distance_to_river_m_log1p", "population_density_per_km2_log1p",
    "rainfall_7d_mm_log1p", "monthly_rainfall_mm_log1p",
    "nearest_hospital_km_log1p", "nearest_evac_km_log1p",
    "elevation_m_yeojohnson", "drainage_index_yeojohnson",
    "ndvi_qmap", "ndwi_qmap", "built_up_percent_qmap",
    "water_presence_flag",
]
features = [c for c in train.columns if c not in DROP_COLS and c in test.columns]
cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]
print(f"Features: {len(features)}  |  Categoricals: {len(cat_cols)}")

X_full = train[features].copy()
y = train[TARGET].astype(float)
X_test_full = test[features].copy()

TE_SOURCE_COLS = [c for c in ["district", "soil_type", "landcover", "urban_rural"] if c in features]


def per_fold_te(tr_vals, va_vals, te_vals, y_tr):
    df_tr = pd.DataFrame({"col": tr_vals, "y": y_tr.values})
    mp = df_tr.groupby("col")["y"].mean()
    gm = y_tr.mean()
    return (
        pd.Series(tr_vals).map(mp).fillna(gm).values,
        pd.Series(va_vals).map(mp).fillna(gm).values,
        pd.Series(te_vals).map(mp).fillna(gm).values,
    )


def encode_train_apply(X_train, X_val, X_test, cols):
    Xt, Xv, Xe = X_train.copy(), X_val.copy(), X_test.copy()
    for c in cols:
        tr = Xt[c].astype(str).fillna("__MISSING__")
        va = Xv[c].astype(str).fillna("__MISSING__")
        te = Xe[c].astype(str).fillna("__MISSING__")
        le = LabelEncoder()
        le.fit(list(tr.unique()) + ["__UNSEEN__"])
        known = set(le.classes_)
        va = va.where(va.isin(known), "__UNSEEN__")
        te = te.where(te.isin(known), "__UNSEEN__")
        Xt[c] = le.transform(tr); Xv[c] = le.transform(va); Xe[c] = le.transform(te)
    return Xt, Xv, Xe


lgb_params = dict(objective="regression", metric="rmse", learning_rate=0.03,
                  num_leaves=127, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=5, min_data_in_leaf=30, lambda_l1=0.1,
                  lambda_l2=0.1, verbose=-1, seed=SEED)

xgb_params = dict(objective="reg:squarederror", eval_metric="rmse",
                  learning_rate=0.03, max_depth=6, min_child_weight=30,
                  subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                  reg_lambda=0.1, n_estimators=3000,
                  early_stopping_rounds=100, verbosity=0, random_state=SEED)

# (A) Tuned CatBoost params from Optuna (60 trials, 3-fold CV, OOF RMSE=0.23526)
cat_params = dict(
    loss_function="RMSE",
    learning_rate=0.04816260673412846,
    depth=5,
    l2_leaf_reg=3.0250412944125444,
    bagging_temperature=0.909674891937747,
    random_strength=2.5943417914394113,
    border_count=40,
    min_data_in_leaf=5,
    iterations=3000,
    early_stopping_rounds=100,
    random_seed=SEED,
    verbose=0,
    task_type="GPU",
    devices="0",
)

gkf = GroupKFold(n_splits=N_FOLDS)
oof_lgb = np.zeros(len(X_full))
oof_xgb = np.zeros(len(X_full))
oof_cat = np.zeros(len(X_full))
preds_lgb = np.zeros(len(X_test_full))
preds_xgb = np.zeros(len(X_test_full))
preds_cat = np.zeros(len(X_test_full))

print("\nTraining...")
for fold, (tr, va) in enumerate(gkf.split(X_full, y, groups=spatial_groups), 1):
    Xtr_raw = X_full.iloc[tr].copy()
    Xva_raw = X_full.iloc[va].copy()
    Xte_raw = X_test_full.copy()
    ytr, yva = y.iloc[tr], y.iloc[va]

    for col in TE_SOURCE_COLS:
        tr_te, va_te, te_te = per_fold_te(
            Xtr_raw[col].values, Xva_raw[col].values, Xte_raw[col].values, ytr,
        )
        Xtr_raw[f"{col}_te"] = tr_te
        Xva_raw[f"{col}_te"] = va_te
        Xte_raw[f"{col}_te"] = te_te

    Xtr_l, Xva_l, Xte_l = Xtr_raw.copy(), Xva_raw.copy(), Xte_raw.copy()
    for c in cat_cols:
        Xtr_l[c] = Xtr_l[c].astype("category")
        Xva_l[c] = pd.Categorical(Xva_l[c], categories=Xtr_l[c].cat.categories)
        Xte_l[c] = pd.Categorical(Xte_l[c], categories=Xtr_l[c].cat.categories)
    dtr = lgb.Dataset(Xtr_l, ytr, categorical_feature=cat_cols)
    dva = lgb.Dataset(Xva_l, yva, categorical_feature=cat_cols)
    m = lgb.train(lgb_params, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof_lgb[va] = m.predict(Xva_l, num_iteration=m.best_iteration)
    preds_lgb += m.predict(Xte_l, num_iteration=m.best_iteration) / N_FOLDS

    Xtr_e, Xva_e, Xte_e = encode_train_apply(Xtr_raw, Xva_raw, Xte_raw, cat_cols)

    m = xgb.XGBRegressor(**xgb_params)
    m.fit(Xtr_e, ytr, eval_set=[(Xva_e, yva)], verbose=False)
    oof_xgb[va] = m.predict(Xva_e)
    preds_xgb += m.predict(Xte_e) / N_FOLDS

    cat_idx = [Xtr_e.columns.tolist().index(c) for c in cat_cols if c in Xtr_e.columns]
    m = cb.CatBoostRegressor(**cat_params, cat_features=cat_idx)
    m.fit(Xtr_e, ytr, eval_set=(Xva_e, yva), use_best_model=True)
    oof_cat[va] = m.predict(Xva_e)
    preds_cat += m.predict(Xte_e) / N_FOLDS

    print(
        f"  fold {fold}  "
        f"LGB={np.sqrt(mean_squared_error(yva, oof_lgb[va])):.5f}  "
        f"XGB={np.sqrt(mean_squared_error(yva, oof_xgb[va])):.5f}  "
        f"CAT={np.sqrt(mean_squared_error(yva, oof_cat[va])):.5f}"
    )

# ─── Optimal weight blend ────────────────────────────────────────────────────
def blend_rmse(w):
    w = np.array(w) / np.array(w).sum()
    return np.sqrt(mean_squared_error(y, w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat))

res = minimize(blend_rmse, x0=[1/3, 1/3, 1/3], method="Nelder-Mead", options={"maxiter": 1000, "xatol": 1e-6})
w = np.array(res.x); w = w / w.sum()
oof_blend = w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat
test_blend = w[0]*preds_lgb + w[1]*preds_xgb + w[2]*preds_cat

print(f"\nBlend weights: LGB={w[0]:.3f} XGB={w[1]:.3f} CAT={w[2]:.3f}")
print(f"Blend OOF RMSE:   {np.sqrt(mean_squared_error(y, oof_blend)):.5f}")
print(f"Blend OOF custom: {custom_metric(y, oof_blend):.5f}")
print(f"Pred std: {oof_blend.std():.4f}  |  Target std: {y.std():.4f}  |  Ratio: {y.std()/oof_blend.std():.2f}x")

# ─── Calibration comparison ──────────────────────────────────────────────────
print("\n" + "="*70)
print("Calibration comparison")
print("="*70)

# (B) Isotonic regression — fit on full OOF (no leakage: each OOF pred is out-of-fold)
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(oof_blend, y)
oof_iso = iso.predict(oof_blend)
test_iso = np.clip(iso.predict(test_blend), 0, 1)

iso_rmse = np.sqrt(mean_squared_error(y, oof_iso))
iso_r2 = r2_score(y, oof_iso)
iso_custom = custom_metric(y, oof_iso)

print(f"\nIsotonic regression calibration (OOF):")
print(f"  RMSE:   {iso_rmse:.5f}")
print(f"  R²:     {iso_r2:.4f}")
print(f"  Custom: {iso_custom:.5f}")
print(f"  Std:    {oof_iso.std():.4f}")

# Linear scale factor (v5 method) for comparison
mean_pred = oof_blend.mean()

def calibrate_linear(arr, factor):
    return np.clip(mean_pred + (arr - mean_pred) * factor, 0, 1)

res_lin = minimize_scalar(
    lambda f: custom_metric(y, calibrate_linear(oof_blend, f)),
    bounds=(0.5, 8.0), method="bounded", options={"xatol": 1e-3}
)
best_factor = res_lin.x
oof_linear = calibrate_linear(oof_blend, best_factor)
test_linear = calibrate_linear(test_blend, best_factor)

lin_rmse = np.sqrt(mean_squared_error(y, oof_linear))
lin_r2 = r2_score(y, oof_linear)
lin_custom = custom_metric(y, oof_linear)

print(f"\nLinear scale calibration (v5 method, factor={best_factor:.3f}):")
print(f"  RMSE:   {lin_rmse:.5f}")
print(f"  R²:     {lin_r2:.4f}")
print(f"  Custom: {lin_custom:.5f}")
print(f"  Std:    {oof_linear.std():.4f}")

print(f"\nCalibration winner: {'isotonic' if iso_custom < lin_custom else 'linear'}")
print(f"  Isotonic vs linear custom metric: {iso_custom:.5f} vs {lin_custom:.5f}")

# ─── Write submissions ───────────────────────────────────────────────────────
# Primary: isotonic calibration
sub_iso = pd.DataFrame({ID: test[ID], TARGET: test_iso})
sub_iso.to_csv(OUT / "improved_ensemble_v6_isotonic.csv", index=False)
print(f"\nWrote {OUT / 'improved_ensemble_v6_isotonic.csv'}")

# Secondary: linear calibration with tuned CatBoost (for ablation)
sub_lin = pd.DataFrame({ID: test[ID], TARGET: np.clip(test_linear, 0, 1)})
sub_lin.to_csv(OUT / "improved_ensemble_v6_linear.csv", index=False)
print(f"Wrote {OUT / 'improved_ensemble_v6_linear.csv'}")

print("\nDone.")
