"""Tier 1 improvements over improved_model.py:

(A) Per-fold target encoding — no leakage of validation targets into TE means.
(A) LabelEncoder fit on train only, unseen test categories mapped to a sentinel.
(B) Spatial GroupKFold using KMeans clusters on lat/lon (NOT as a feature).
(C) Missingness flags computed BEFORE any fillna.
(D) DROP_COLS cleanup — remove pre-engineered _log1p/_yeojohnson/_qmap duplicates
    and the highly redundant water_presence_flag.

Run:
    python src/improved_model_v2.py
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
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
N_SPATIAL_CLUSTERS = 20  # for GroupKFold splitting

# ─── Load ───────────────────────────────────────────────────────────────────
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
for df in (train, test):
    df["nearest_evac_km"] = pd.to_numeric(df["nearest_evac_km"], errors="coerce")

print(f"Train: {train.shape}  Test: {test.shape}")

# ─── (C) Missingness flags — BEFORE any fillna ──────────────────────────────
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
print(f"Added {len(MISSING_FLAG_COLS)} missingness flags.")

# ─── Interaction features (kept from v1) ────────────────────────────────────
def add_interactions(df):
    df = df.copy()
    df["rain_x_flood_count"] = (
        df["rainfall_7d_mm"].fillna(0) * df["historical_flood_count"].fillna(0)
    )
    df["elevation_x_river"] = (
        df["elevation_m"].fillna(0) * (1 / (df["distance_to_river_m"].fillna(1000) + 1))
    )
    df["rain_ratio"] = (
        df["rainfall_7d_mm"].fillna(0) / (df["monthly_rainfall_mm"].fillna(1) + 1)
    )
    df["risk_density"] = (
        df["historical_flood_count"].fillna(0) * df["population_density_per_km2"].fillna(0)
    )
    df["infra_x_evac"] = (
        df["infrastructure_score"].fillna(50) / (df["nearest_evac_km"].fillna(10) + 1)
    )
    df["ndwi_x_rain"] = df["ndwi"].fillna(0) * df["rainfall_7d_mm"].fillna(0)
    df["low_elevation_flag"] = (df["elevation_m"].fillna(100) < 10).astype(int)
    df["close_river_flag"] = (df["distance_to_river_m"].fillna(9999) < 500).astype(int)
    return df

train = add_interactions(train)
test = add_interactions(test)

# ─── (B) Spatial groups — for fold splitting ONLY (not as feature) ──────────
coords_train = train[["latitude", "longitude"]].fillna(
    train[["latitude", "longitude"]].median()
)
km = KMeans(n_clusters=N_SPATIAL_CLUSTERS, random_state=SEED, n_init=10)
spatial_groups = km.fit_predict(coords_train)
cluster_sizes = pd.Series(spatial_groups).value_counts().sort_index()
print(f"Spatial clusters: {N_SPATIAL_CLUSTERS}  | size range: "
      f"{cluster_sizes.min()}–{cluster_sizes.max()}  | mean: {cluster_sizes.mean():.0f}")

# ─── (D) DROP_COLS cleanup ──────────────────────────────────────────────────
DROP_COLS = [
    ID, TARGET, "generation_date", "place_name", "reason_not_good_to_live",
    "is_good_to_live", "is_synthetic", "flood_occurrence_current_event",
    "inundation_area_sqm",
    # pre-engineered duplicates of raw columns
    "distance_to_river_m_log1p",
    "population_density_per_km2_log1p",
    "rainfall_7d_mm_log1p",
    "monthly_rainfall_mm_log1p",
    "nearest_hospital_km_log1p",
    "nearest_evac_km_log1p",
    "elevation_m_yeojohnson",
    "drainage_index_yeojohnson",
    "ndvi_qmap",
    "ndwi_qmap",
    "built_up_percent_qmap",
    "water_presence_flag",  # r=0.679 with ndwi
]
features = [c for c in train.columns if c not in DROP_COLS and c in test.columns]
cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]
print(f"Features: {len(features)}  |  Categoricals: {len(cat_cols)}")

X_full = train[features].copy()
y = train[TARGET].astype(float)
X_test_full = test[features].copy()

# ─── (A) Per-fold target encoding helper ────────────────────────────────────
TE_SOURCE_COLS = ["district", "soil_type", "landcover", "urban_rural"]
TE_SOURCE_COLS = [c for c in TE_SOURCE_COLS if c in features]

def per_fold_te(tr_vals, va_vals, te_vals, y_tr):
    df_tr = pd.DataFrame({"col": tr_vals, "y": y_tr.values})
    mp = df_tr.groupby("col")["y"].mean()
    gm = y_tr.mean()
    return (
        pd.Series(tr_vals).map(mp).fillna(gm).values,
        pd.Series(va_vals).map(mp).fillna(gm).values,
        pd.Series(te_vals).map(mp).fillna(gm).values,
    )

# ─── (A) Train-only LabelEncoder with unseen-test handling ──────────────────
def encode_train_apply(X_train, X_val, X_test, cols):
    Xt = X_train.copy()
    Xv = X_val.copy()
    Xe = X_test.copy()
    for c in cols:
        tr = Xt[c].astype(str).fillna("__MISSING__")
        va = Xv[c].astype(str).fillna("__MISSING__")
        te = Xe[c].astype(str).fillna("__MISSING__")
        le = LabelEncoder()
        le.fit(list(tr.unique()) + ["__UNSEEN__"])
        known = set(le.classes_)
        va = va.where(va.isin(known), "__UNSEEN__")
        te = te.where(te.isin(known), "__UNSEEN__")
        Xt[c] = le.transform(tr)
        Xv[c] = le.transform(va)
        Xe[c] = le.transform(te)
    return Xt, Xv, Xe

# ─── Hyperparameters (same as v1 for apples-to-apples) ──────────────────────
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

# ─── Spatial CV loop ────────────────────────────────────────────────────────
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

    # (A) Per-fold target encoding on training fold only
    for col in TE_SOURCE_COLS:
        tr_te, va_te, te_te = per_fold_te(
            Xtr_raw[col].values, Xva_raw[col].values, Xte_raw[col].values, ytr,
        )
        Xtr_raw[f"{col}_te"] = tr_te
        Xva_raw[f"{col}_te"] = va_te
        Xte_raw[f"{col}_te"] = te_te

    # ── LightGBM with native categorical handling ──
    Xtr_l = Xtr_raw.copy()
    Xva_l = Xva_raw.copy()
    Xte_l = Xte_raw.copy()
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

    # ── Label-encode (train-only fit) for XGB / CatBoost ──
    Xtr_e, Xva_e, Xte_e = encode_train_apply(Xtr_raw, Xva_raw, Xte_raw, cat_cols)

    # ── XGBoost ──
    m = xgb.XGBRegressor(**xgb_params)
    m.fit(Xtr_e, ytr, eval_set=[(Xva_e, yva)], verbose=False)
    oof_xgb[va] = m.predict(Xva_e)
    preds_xgb += m.predict(Xte_e) / N_FOLDS

    # ── CatBoost ──
    cat_idx = [Xtr_e.columns.tolist().index(c) for c in cat_cols if c in Xtr_e.columns]
    m = cb.CatBoostRegressor(**cat_params, cat_features=cat_idx)
    m.fit(Xtr_e, ytr, eval_set=(Xva_e, yva), use_best_model=True)
    oof_cat[va] = m.predict(Xva_e)
    preds_cat += m.predict(Xte_e) / N_FOLDS

    print(
        f"  fold {fold}  "
        f"LGB={np.sqrt(mean_squared_error(yva, oof_lgb[va])):.5f}  "
        f"XGB={np.sqrt(mean_squared_error(yva, oof_xgb[va])):.5f}  "
        f"CAT={np.sqrt(mean_squared_error(yva, oof_cat[va])):.5f}  "
        f"(val_size={len(va)})"
    )

print()
lgb_rmse = np.sqrt(mean_squared_error(y, oof_lgb))
xgb_rmse = np.sqrt(mean_squared_error(y, oof_xgb))
cat_rmse = np.sqrt(mean_squared_error(y, oof_cat))
print(f"OOF RMSE  LGB: {lgb_rmse:.5f}")
print(f"OOF RMSE  XGB: {xgb_rmse:.5f}")
print(f"OOF RMSE  CAT: {cat_rmse:.5f}")

# ─── Blending ───────────────────────────────────────────────────────────────
def blend_rmse(w):
    w = np.array(w) / np.array(w).sum()
    return np.sqrt(mean_squared_error(y, w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat))

res = minimize(blend_rmse, x0=[1/3, 1/3, 1/3], method="Nelder-Mead",
               options={"maxiter": 1000, "xatol": 1e-6})
w = np.array(res.x); w = w / w.sum()
simple_rmse = np.sqrt(mean_squared_error(y, (oof_lgb + oof_xgb + oof_cat) / 3))
optimal_rmse = blend_rmse(w)

print(f"\nBlend weights — LGB:{w[0]:+.3f}  XGB:{w[1]:+.3f}  CAT:{w[2]:+.3f}")
print(f"Simple avg OOF:  {simple_rmse:.5f}")
print(f"Optimal blend:   {optimal_rmse:.5f}")

print()
print("="*70)
print("REFERENCE (random KFold, leaky TE)")
print("="*70)
print(f"  v1 baseline LightGBM:           0.23585")
print(f"  v1 ensemble (random KFold):     0.23511   (LB 0.38504)")
print(f"\nv2 OOF should be HIGHER (this is HONEST — closer to actual LB).")
print(f"Lower OOF-vs-LB gap = better generalisation.")

# ─── Write submission file (NOT uploaded) ───────────────────────────────────
final_preds = w[0]*preds_lgb + w[1]*preds_xgb + w[2]*preds_cat
final_preds = np.clip(final_preds, 0, 1)
sub = pd.DataFrame({ID: test[ID], TARGET: final_preds})
sub.to_csv(OUT / "improved_ensemble_v2.csv", index=False)
print(f"\nWrote {OUT / 'improved_ensemble_v2.csv'}  (NOT uploaded)")
