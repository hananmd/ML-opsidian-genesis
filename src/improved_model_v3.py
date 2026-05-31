"""v3: v2 + Tier 2 features.

New in v3:
(E) District-level statistical features (computed on train only).
(F) Relative features — value minus district mean.
(G) Compound 3-variable features (catchment_saturation proxy from drainage_index).
(H) DBSCAN isolation flag.

Run:
    python src/improved_model_v3.py
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
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

# ─── Load ───────────────────────────────────────────────────────────────────
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
for df in (train, test):
    df["nearest_evac_km"] = pd.to_numeric(df["nearest_evac_km"], errors="coerce")
print(f"Train: {train.shape}  Test: {test.shape}")

# ─── (C) Missingness flags BEFORE any fillna ────────────────────────────────
MISSING_FLAG_COLS = [
    "electricity", "road_quality", "distance_to_river_m",
    "drainage_index", "ndvi", "ndwi", "infrastructure_score",
    "nearest_evac_km", "nearest_hospital_km", "soil_type",
    "population_density_per_km2", "built_up_percent",
]
for df in (train, test):
    for c in MISSING_FLAG_COLS:
        if c in df.columns:
            df[f"{c}_missing"] = df[c].isnull().astype(int)

# ─── Interaction features ───────────────────────────────────────────────────
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

# ─── (E) District-level statistical features (train-only mapping) ───────────
DIST_STAT_COLS = [
    "historical_flood_count", "rainfall_7d_mm", "monthly_rainfall_mm",
    "infrastructure_score", "elevation_m", "distance_to_river_m",
    "drainage_index", "ndwi", "population_density_per_km2",
]
print(f"\nAdding district-level statistics ({len(DIST_STAT_COLS)} columns × mean/std)...")
for col in DIST_STAT_COLS:
    if col not in train.columns:
        continue
    grp = train.groupby("district")[col]
    mean_map = grp.mean()
    std_map = grp.std()
    global_mean = train[col].mean()
    global_std = train[col].std()
    for df in (train, test):
        df[f"district_{col}_mean"] = df["district"].map(mean_map).fillna(global_mean)
        df[f"district_{col}_std"] = df["district"].map(std_map).fillna(global_std)

# ─── (F) Relative features ──────────────────────────────────────────────────
RELATIVE_PAIRS = [
    ("elevation_m", "district_elevation_m_mean"),
    ("distance_to_river_m", "district_distance_to_river_m_mean"),
    ("rainfall_7d_mm", "district_rainfall_7d_mm_mean"),
    ("infrastructure_score", "district_infrastructure_score_mean"),
    ("drainage_index", "district_drainage_index_mean"),
]
print("Adding relative features...")
for raw, ref in RELATIVE_PAIRS:
    if raw in train.columns and ref in train.columns:
        for df in (train, test):
            df[f"rel_{raw}"] = df[raw] - df[ref]

# Z-score for rainfall (most signal)
for df in (train, test):
    df["rainfall_district_zscore"] = (
        (df["rainfall_7d_mm"] - df["district_rainfall_7d_mm_mean"])
        / (df["district_rainfall_7d_mm_std"] + 0.01)
    )

# ─── (G) Compound 3-variable risk features (catchment_saturation proxy) ─────
# Proxy: high monthly_rainfall + low drainage_index => high saturation
# drainage_index here is normalised — higher = better drainage
# Saturation proxy = monthly_rainfall * (1 - normalised drainage)
print("Adding compound risk features...")
drain_norm = (train["drainage_index"].fillna(train["drainage_index"].median()) -
              train["drainage_index"].min()) / (train["drainage_index"].max() - train["drainage_index"].min() + 1e-9)
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

# ─── (H) DBSCAN isolation flag ──────────────────────────────────────────────
print("Computing DBSCAN isolation...")
all_coords = pd.concat([train[["latitude", "longitude"]], test[["latitude", "longitude"]]])
all_coords = all_coords.fillna(all_coords.median())
eps_radians = 3.0 / 6371.0  # 3 km radius on the Earth
dbscan = DBSCAN(eps=eps_radians, min_samples=20, algorithm="ball_tree", metric="haversine")
labels = dbscan.fit_predict(np.radians(all_coords.values))
train["is_isolated_location"] = (labels[:len(train)] == -1).astype(int)
test["is_isolated_location"] = (labels[len(train):] == -1).astype(int)
print(f"  Isolated: {train['is_isolated_location'].sum()} train, {test['is_isolated_location'].sum()} test")

# ─── (B) Spatial groups for fold splitting ──────────────────────────────────
coords_train = train[["latitude", "longitude"]].fillna(train[["latitude", "longitude"]].median())
km = KMeans(n_clusters=N_SPATIAL_CLUSTERS, random_state=SEED, n_init=10)
spatial_groups = km.fit_predict(coords_train)

# ─── (D) DROP_COLS cleanup ──────────────────────────────────────────────────
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
print(f"\nTotal features: {len(features)}  |  Categoricals: {len(cat_cols)}")

X_full = train[features].copy()
y = train[TARGET].astype(float)
X_test_full = test[features].copy()

# ─── (A) Per-fold target encoding ───────────────────────────────────────────
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

# ─── Hyperparameters ────────────────────────────────────────────────────────
lgb_params = dict(
    objective="regression", metric="rmse", learning_rate=0.03,
    num_leaves=127, feature_fraction=0.8, bagging_fraction=0.8,
    bagging_freq=5, min_data_in_leaf=30,
    lambda_l1=0.1, lambda_l2=0.1, verbose=-1, seed=SEED,
)
xgb_params = dict(
    objective="reg:squarederror", eval_metric="rmse", learning_rate=0.03,
    max_depth=6, min_child_weight=30, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, n_estimators=3000,
    early_stopping_rounds=100, verbosity=0, random_state=SEED,
)
cat_params = dict(
    loss_function="RMSE", learning_rate=0.03, depth=6, l2_leaf_reg=3,
    iterations=3000, early_stopping_rounds=100, random_seed=SEED, verbose=0,
)

# ─── CV loop ────────────────────────────────────────────────────────────────
gkf = GroupKFold(n_splits=N_FOLDS)
oof_lgb = np.zeros(len(X_full))
oof_xgb = np.zeros(len(X_full))
oof_cat = np.zeros(len(X_full))
preds_lgb = np.zeros(len(X_test_full))
preds_xgb = np.zeros(len(X_test_full))
preds_cat = np.zeros(len(X_test_full))

print("\n" + "="*70)
print(f"Spatial GroupKFold (5 folds on {N_SPATIAL_CLUSTERS} KMeans clusters)")
print("="*70)

for fold, (tr, va) in enumerate(gkf.split(X_full, y, groups=spatial_groups), 1):
    Xtr_raw = X_full.iloc[tr].copy()
    Xva_raw = X_full.iloc[va].copy()
    Xte_raw = X_test_full.copy()
    ytr, yva = y.iloc[tr], y.iloc[va]

    # Per-fold target encoding
    for col in TE_SOURCE_COLS:
        tr_te, va_te, te_te = per_fold_te(
            Xtr_raw[col].values, Xva_raw[col].values, Xte_raw[col].values, ytr,
        )
        Xtr_raw[f"{col}_te"] = tr_te
        Xva_raw[f"{col}_te"] = va_te
        Xte_raw[f"{col}_te"] = te_te

    # LightGBM
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

    # Encode for XGB/CatBoost
    Xtr_e, Xva_e, Xte_e = encode_train_apply(Xtr_raw, Xva_raw, Xte_raw, cat_cols)

    # XGBoost
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(Xtr_e, ytr, eval_set=[(Xva_e, yva)], verbose=False)
    oof_xgb[va] = m.predict(Xva_e)
    preds_xgb += m.predict(Xte_e) / N_FOLDS

    # CatBoost
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

print()
print(f"OOF RMSE  LGB: {np.sqrt(mean_squared_error(y, oof_lgb)):.5f}")
print(f"OOF RMSE  XGB: {np.sqrt(mean_squared_error(y, oof_xgb)):.5f}")
print(f"OOF RMSE  CAT: {np.sqrt(mean_squared_error(y, oof_cat)):.5f}")

# ─── Blend ──────────────────────────────────────────────────────────────────
def blend_rmse(w):
    w = np.array(w) / np.array(w).sum()
    return np.sqrt(mean_squared_error(y, w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat))

res = minimize(blend_rmse, x0=[1/3, 1/3, 1/3], method="Nelder-Mead",
               options={"maxiter": 1000, "xatol": 1e-6})
w = np.array(res.x); w = w / w.sum()
simple = np.sqrt(mean_squared_error(y, (oof_lgb + oof_xgb + oof_cat) / 3))
optimal = blend_rmse(w)

print(f"\nBlend  LGB:{w[0]:+.3f}  XGB:{w[1]:+.3f}  CAT:{w[2]:+.3f}")
print(f"Simple avg OOF:  {simple:.5f}")
print(f"Optimal blend:   {optimal:.5f}")

print("\n" + "="*70)
print("History (RMSE):")
print("  v0 baseline LGB:           0.23585    (LB 0.38699)")
print("  v1 ensemble (random KF):   0.23511    (LB 0.38504)")
print("  v2 spatial + per-fold TE:  0.23511    (LB 0.38510)")
print(f"  v3 + Tier 2 features:      {optimal:.5f}")

final_preds = w[0]*preds_lgb + w[1]*preds_xgb + w[2]*preds_cat
final_preds = np.clip(final_preds, 0, 1)
sub = pd.DataFrame({ID: test[ID], TARGET: final_preds})
sub.to_csv(OUT / "improved_ensemble_v3.csv", index=False)
print(f"\nWrote {OUT / 'improved_ensemble_v3.csv'}  (NOT uploaded)")
