"""v4 = v3 + PyTorch MLP added to ensemble (GPU).

Models in blend: LightGBM + XGBoost + CatBoost + MLP.
The MLP captures non-tree patterns and adds diversity.

Run:
    python src/improved_model_v4.py
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.optimize import minimize
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "submissions"
OUT.mkdir(exist_ok=True)

TARGET = "flood_risk_score"
ID = "record_id"
SEED = 42
N_FOLDS = 5
N_SPATIAL_CLUSTERS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

torch.manual_seed(SEED)
np.random.seed(SEED)

# ─── Load ───────────────────────────────────────────────────────────────────
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
for df in (train, test):
    df["nearest_evac_km"] = pd.to_numeric(df["nearest_evac_km"], errors="coerce")
print(f"Train: {train.shape}  Test: {test.shape}")

# ─── (C) Missingness flags before fillna ────────────────────────────────────
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

# ─── (E) District-level stats ───────────────────────────────────────────────
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

# ─── (F) Relative features ──────────────────────────────────────────────────
RELATIVE_PAIRS = [
    ("elevation_m", "district_elevation_m_mean"),
    ("distance_to_river_m", "district_distance_to_river_m_mean"),
    ("rainfall_7d_mm", "district_rainfall_7d_mm_mean"),
    ("infrastructure_score", "district_infrastructure_score_mean"),
    ("drainage_index", "district_drainage_index_mean"),
]
for raw, ref in RELATIVE_PAIRS:
    if raw in train.columns and ref in train.columns:
        for df in (train, test):
            df[f"rel_{raw}"] = df[raw] - df[ref]

for df in (train, test):
    df["rainfall_district_zscore"] = (
        (df["rainfall_7d_mm"] - df["district_rainfall_7d_mm_mean"])
        / (df["district_rainfall_7d_mm_std"] + 0.01)
    )

# ─── (G) Compound risk features ─────────────────────────────────────────────
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

# ─── (H) DBSCAN isolation ───────────────────────────────────────────────────
all_coords = pd.concat([train[["latitude", "longitude"]], test[["latitude", "longitude"]]])
all_coords = all_coords.fillna(all_coords.median())
dbscan = DBSCAN(eps=10.0 / 6371.0, min_samples=10, algorithm="ball_tree", metric="haversine")
labels = dbscan.fit_predict(np.radians(all_coords.values))
train["is_isolated_location"] = (labels[:len(train)] == -1).astype(int)
test["is_isolated_location"] = (labels[len(train):] == -1).astype(int)

# ─── Spatial groups ─────────────────────────────────────────────────────────
coords_train = train[["latitude", "longitude"]].fillna(train[["latitude", "longitude"]].median())
spatial_groups = KMeans(n_clusters=N_SPATIAL_CLUSTERS, random_state=SEED, n_init=10).fit_predict(coords_train)

# ─── Drop unused ────────────────────────────────────────────────────────────
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

# ─── MLP architecture ───────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, n_in, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]  # target is in [0, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_tr, y_tr, X_va, y_va, X_te,
              n_epochs=200, batch_size=512, lr=1e-3, patience=20):
    """Train MLP with early stopping. Returns (val_preds, test_preds)."""
    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    ytr = torch.tensor(y_tr.values, dtype=torch.float32, device=DEVICE)
    Xva = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    yva = torch.tensor(y_va.values, dtype=torch.float32, device=DEVICE)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)

    model = MLP(X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    n = Xtr.shape[0]
    best_val = float("inf")
    best_state = None
    patience_left = patience

    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            preds = model(Xtr[idx])
            loss = loss_fn(preds, ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            v_loss = loss_fn(model(Xva), yva).item()
        if v_loss < best_val - 1e-6:
            best_val = v_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_pred = model(Xva).cpu().numpy()
        te_pred = model(Xte).cpu().numpy()
    return va_pred, te_pred, np.sqrt(best_val), epoch


# ─── Hyperparameters for trees ──────────────────────────────────────────────
lgb_params = dict(objective="regression", metric="rmse", learning_rate=0.03,
                  num_leaves=127, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=5, min_data_in_leaf=30, lambda_l1=0.1,
                  lambda_l2=0.1, verbose=-1, seed=SEED)
xgb_params = dict(objective="reg:squarederror", eval_metric="rmse",
                  learning_rate=0.03, max_depth=6, min_child_weight=30,
                  subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                  reg_lambda=0.1, n_estimators=3000,
                  early_stopping_rounds=100, verbosity=0, random_state=SEED)
cat_params = dict(loss_function="RMSE", learning_rate=0.03, depth=6,
                  l2_leaf_reg=3, iterations=3000, early_stopping_rounds=100,
                  random_seed=SEED, verbose=0)

# ─── CV ─────────────────────────────────────────────────────────────────────
gkf = GroupKFold(n_splits=N_FOLDS)
oof_lgb = np.zeros(len(X_full))
oof_xgb = np.zeros(len(X_full))
oof_cat = np.zeros(len(X_full))
oof_mlp = np.zeros(len(X_full))
preds_lgb = np.zeros(len(X_test_full))
preds_xgb = np.zeros(len(X_test_full))
preds_cat = np.zeros(len(X_test_full))
preds_mlp = np.zeros(len(X_test_full))

print("\n" + "="*70)
print(f"Spatial GroupKFold (5 folds on {N_SPATIAL_CLUSTERS} KMeans clusters)")
print("="*70)

for fold, (tr, va) in enumerate(gkf.split(X_full, y, groups=spatial_groups), 1):
    Xtr_raw = X_full.iloc[tr].copy()
    Xva_raw = X_full.iloc[va].copy()
    Xte_raw = X_test_full.copy()
    ytr, yva = y.iloc[tr], y.iloc[va]

    # Per-fold TE
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

    # Label-encode for the other models
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

    # MLP — needs full imputation + scaling
    Xtr_n = Xtr_e.fillna(Xtr_e.median(numeric_only=True)).fillna(0).astype(np.float32)
    Xva_n = Xva_e.fillna(Xtr_e.median(numeric_only=True)).fillna(0).astype(np.float32)
    Xte_n = Xte_e.fillna(Xtr_e.median(numeric_only=True)).fillna(0).astype(np.float32)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr_n.values)
    Xva_s = scaler.transform(Xva_n.values)
    Xte_s = scaler.transform(Xte_n.values)
    va_p, te_p, mlp_rmse, mlp_epochs = train_mlp(Xtr_s, ytr, Xva_s, yva, Xte_s)
    oof_mlp[va] = va_p
    preds_mlp += te_p / N_FOLDS

    print(
        f"  fold {fold}  "
        f"LGB={np.sqrt(mean_squared_error(yva, oof_lgb[va])):.5f}  "
        f"XGB={np.sqrt(mean_squared_error(yva, oof_xgb[va])):.5f}  "
        f"CAT={np.sqrt(mean_squared_error(yva, oof_cat[va])):.5f}  "
        f"MLP={mlp_rmse:.5f} (ep{mlp_epochs})"
    )

print()
print(f"OOF RMSE  LGB: {np.sqrt(mean_squared_error(y, oof_lgb)):.5f}")
print(f"OOF RMSE  XGB: {np.sqrt(mean_squared_error(y, oof_xgb)):.5f}")
print(f"OOF RMSE  CAT: {np.sqrt(mean_squared_error(y, oof_cat)):.5f}")
print(f"OOF RMSE  MLP: {np.sqrt(mean_squared_error(y, oof_mlp)):.5f}")

# ─── Blend (4 models) ───────────────────────────────────────────────────────
def blend_rmse(w):
    w = np.array(w) / (np.array(w).sum() + 1e-9)
    p = w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat + w[3]*oof_mlp
    return np.sqrt(mean_squared_error(y, p))

res = minimize(blend_rmse, x0=[0.25, 0.25, 0.25, 0.25], method="Nelder-Mead",
               options={"maxiter": 2000, "xatol": 1e-6})
w = np.array(res.x); w = w / w.sum()
simple = np.sqrt(mean_squared_error(y, (oof_lgb + oof_xgb + oof_cat + oof_mlp) / 4))
optimal = blend_rmse(w)

# Also compute trees-only blend (3 models, exclude MLP)
def blend3(w):
    w = np.array(w) / w.sum()
    return np.sqrt(mean_squared_error(y, w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat))
res3 = minimize(blend3, x0=[1/3, 1/3, 1/3], method="Nelder-Mead")
w3 = np.array(res3.x) / np.sum(res3.x)
trees_only_rmse = blend3(w3)

print(f"\nBlend (4) LGB:{w[0]:+.3f}  XGB:{w[1]:+.3f}  CAT:{w[2]:+.3f}  MLP:{w[3]:+.3f}")
print(f"Simple avg (4-model):   {simple:.5f}")
print(f"Optimal blend (4):      {optimal:.5f}")
print(f"Trees-only blend (3):   {trees_only_rmse:.5f}")
print(f"MLP contribution:       {trees_only_rmse - optimal:+.5f}")

print("\n" + "="*70)
print("History:")
print("  v0 baseline LGB:           OOF 0.23585  LB 0.38699")
print("  v1 ensemble (random KF):   OOF 0.23511  LB 0.38504")
print("  v2 spatial + per-fold TE:  OOF 0.23511  LB 0.38510")
print("  v3 + Tier 2 features:      OOF 0.23508  LB 0.38488")
print(f"  v4 + MLP in ensemble:      OOF {optimal:.5f}  LB ?")

final_preds = (w[0]*preds_lgb + w[1]*preds_xgb + w[2]*preds_cat + w[3]*preds_mlp)
final_preds = np.clip(final_preds, 0, 1)
sub = pd.DataFrame({ID: test[ID], TARGET: final_preds})
sub.to_csv(OUT / "improved_ensemble_v4.csv", index=False)
print(f"\nWrote {OUT / 'improved_ensemble_v4.csv'}  (NOT uploaded)")
