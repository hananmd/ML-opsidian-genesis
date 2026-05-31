# Flood Risk Prediction — Full Process Log

## Competition
**ML Opsidian Genesis - Initial Round 26**
Task: Predict `flood_risk_score` (0–1) for 5,300 locations in Sri Lanka.
Metric: Custom score = `(mae + rmse) / 2 * (1 + max(0, 1 - r²))` — penalises both error magnitude AND low R².

---

## Results Summary

| Version | File | Key Change | OOF RMSE | Public LB |
|---|---|---|---|---|
| v0 | `baseline_lgbm.csv` | LightGBM baseline | 0.23585 | 0.38699 |
| v1 | `improved_ensemble.csv` | 3-model ensemble + feature eng | 0.23511 | **0.38504** |
| v2 | `improved_ensemble_v2.csv` | Spatial CV + leak fixes | 0.23511 | 0.38510 |
| v3 | `improved_ensemble_v3.csv` | Tier 2 features | 0.23508 | **0.38488** |
| v4 | `improved_ensemble_v4.csv` | + PyTorch MLP | ? | ? |
| v5 raw | `improved_ensemble_v5_raw.csv` | v3 trees, uncalibrated | ? | ? |
| v5 calibrated | `improved_ensemble_v5_calibrated.csv` | + OOF prediction spread calibration | ? | ? |

**Current confirmed best: v3 → LB 0.38488**

---

## Step 1 — Baseline LightGBM (`src/baseline_lgbm.py`)

- 5-fold random KFold, LightGBM only
- Native categorical handling (no encoding)
- Early stopping at 100 rounds, max 3000 rounds
- Dropped: `record_id`, `generation_date`, `place_name`, `reason_not_good_to_live`
- Clipped predictions to [0, 1]

**Result:** OOF RMSE = 0.23585 | LB = 0.38699

---

## Step 2 — EDA (`notebooks/eda.py`)

Key findings that shaped all subsequent models:

- Target well-distributed (mean 0.478, std 0.239) — no class imbalance
- All numeric features have near-zero **linear** correlation with target — relationships are non-linear
- Top predictors (Random Forest importance): `extreme_weather_index`, `terrain_roughness_index`, `district`, `distance_to_river_m`, `socioeconomic_status_index`, `latitude/longitude`
- Missing values ~4–8% across many columns — LightGBM handles natively
- District matters: Vavuniya (mean 0.538) vs Colombo (mean 0.410) — 13% range
- Soil type and landcover have very low signal (<1% variance in mean target)

Plots generated in `notebooks/eda_output/`:
1. Target distribution
2. Missing values by column
3. Numeric correlations with target
4. Categorical features vs target (boxplots)
5. Geographic scatter (lat/lon colored by risk)
6. District-level mean risk
7. Key scatter plots (elevation, distance to river, rainfall, etc.)
8. Random Forest feature importance

---

## Step 3 — Improved Ensemble v1 (`src/improved_model.py`)

### Feature Engineering Added
| Feature | Formula |
|---|---|
| `rain_x_flood_count` | `rainfall_7d_mm × historical_flood_count` |
| `elevation_x_river` | `elevation_m × (1 / distance_to_river_m)` |
| `rain_ratio` | `rainfall_7d_mm / monthly_rainfall_mm` |
| `risk_density` | `historical_flood_count × population_density` |
| `infra_x_evac` | `infrastructure_score / nearest_evac_km` |
| `ndwi_x_rain` | `ndwi × rainfall_7d_mm` |
| `low_elevation_flag` | Binary: elevation < 10m |
| `close_river_flag` | Binary: distance_to_river < 500m |
| `geo_cluster` | KMeans(25) on lat/lon |
| `*_te` | Target encoding for district, soil_type, landcover, urban_rural, geo_cluster |

### Models (5-fold random KFold each)
| Model | OOF RMSE | Notes |
|---|---|---|
| LightGBM | 0.23644 | |
| XGBoost | 0.23555 | |
| CatBoost | **0.23512** | Best single model |

### Ensemble
- Nelder-Mead optimised blend: LGB -11%, XGB 15%, CatBoost 96%
- Simple average OOF RMSE: 0.23541
- Optimal blend OOF RMSE: **0.23511**

**Result:** LB = 0.38504 ✅ (best at the time)

---

## Step 4 — Integrity Fixes + Spatial CV (`src/improved_model_v2.py`)

Motivation: v1 had three sources of data leakage that inflated local CV scores.

### Changes Made

**(A) Per-fold target encoding** — v1 computed TE means across all training data before splitting folds. Fixed to compute TE means on the training fold only, so validation targets never leak into encoding.

**(A) Train-only LabelEncoder** — v1 fit LabelEncoder on train+test combined, leaking test category distributions. Fixed to fit on train only; unseen test categories mapped to `__UNSEEN__` sentinel.

**(B) Spatial GroupKFold** — v1 used random KFold, allowing nearby geographic locations into both train and val (inflating CV scores since neighbors share flood patterns). Fixed to KMeans(20) on lat/lon, then GroupKFold so each fold trains and validates on geographically separated areas.

**(C) Missingness flags** — Added 12 binary flags (`*_missing`) computed **before** any fillna. Missing values in infrastructure columns are not random (missing `road_quality` = no road exists). LightGBM treats all nulls identically; flags preserve this signal.

**(D) DROP_COLS cleanup** — Dataset contained pre-engineered variants of raw columns (`_log1p`, `_yeojohnson`, `_qmap` suffix columns) that duplicate information and encode test distribution. Dropped them all. Also dropped `water_presence_flag` (r=0.679 with ndwi — redundant).

**Result:** OOF RMSE = 0.23511 (same, but now honest) | LB = 0.38510 (slightly worse than v1 — confirms v1 OOF was optimistic due to leakage)

---

## Step 5 — Tier 2 Feature Engineering (`src/improved_model_v3.py`)

### Changes Made

**(E) District-level statistical features** — Instead of relying on district as a raw category, extracted what district encodes as numeric features. Computed on train only, mapped to test via district name.

```
district_{col}_mean  and  district_{col}_std
```
For cols: `historical_flood_count`, `rainfall_7d_mm`, `monthly_rainfall_mm`, `infrastructure_score`, `elevation_m`, `distance_to_river_m`, `drainage_index`, `ndwi`, `population_density_per_km2`.

**(F) Relative features** — Normalised absolute values against district averages. `elevation=150m` means different things in different districts.

```
rel_elevation_m  =  elevation_m - district_elevation_m_mean
rel_distance_to_river_m, rel_rainfall_7d_mm, rel_infrastructure_score, rel_drainage_index
rainfall_district_zscore  =  (rainfall - district_mean) / (district_std + 0.01)
```

**(G) Compound 3-variable risk features** — Captures multi-variable flood risk signal that 2-variable ratios miss. Uses a `catchment_saturation_proxy` = monthly rainfall × (1 - normalised drainage).

```
catchment_saturation_proxy  =  monthly_rainfall_mm × (1 - normalised_drainage)
compound_flood_risk         =  (rainfall_7d_mm × saturation) / (elevation_m + 1)
historical_risk_amplifier   =  (historical_flood_count + 1) × saturation
evacuation_risk             =  nearest_evac_km / (infrastructure_score + 0.1)
triple_risk                 =  rainfall_7d_mm × historical_flood_count × low_elevation_flag
```

**(H) DBSCAN isolation flag** — Flags geographically isolated locations (DBSCAN cluster = -1). KMeans assigns every point to a cluster; DBSCAN specifically identifies outliers. Isolated rural locations have distinct dynamics. Parameters: 3km radius, min_samples=20, haversine metric.

**Result:** OOF RMSE = 0.23508 | LB = **0.38488** ✅ (best confirmed score)

---

## Step 6 — PyTorch MLP Added (`src/improved_model_v4.py`)

Added a 4th model (PyTorch MLP) to the ensemble to capture non-tree patterns and add diversity.

### MLP Architecture
- Input → Linear(256) → BatchNorm → ReLU → Dropout(0.3)
- → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
- → Linear(64) → BatchNorm → ReLU → Dropout(0.3)
- → Linear(1) → Sigmoid (output in [0,1])
- Adam optimiser, lr=1e-3, weight_decay=1e-5
- Early stopping (patience=20 epochs), max 200 epochs
- Median imputation + StandardScaler before MLP

Blend: LGB + XGB + CatBoost + MLP (4-way Nelder-Mead optimised).

Note: v4 used slightly looser DBSCAN params (eps=10km, min_samples=10) vs v3 (eps=3km, min_samples=20).

**Result:** LB = ? (not yet submitted)

---

## Step 7 — Prediction Calibration (`src/improved_model_v5.py`)

### Motivation
The custom metric penalises low R². OOF predictions had std ≈ 0.04 vs target std ≈ 0.24 — predictions were 6× too compressed, killing R². Calibration spreads predictions away from their mean.

### Method
```
calibrated = clip(mean_pred + (pred - mean_pred) × factor, 0, 1)
```
Optimal `factor` found by minimising the custom metric on OOF predictions using `minimize_scalar` (bounded search in [0.5, 8.0]).

A calibration grid was printed for factors 1.0–6.0 to understand the rmse/r²/custom metric tradeoff.

### Outputs
- `improved_ensemble_v5_calibrated.csv` — calibration applied
- `improved_ensemble_v5_raw.csv` — same v3 tree ensemble, no calibration (baseline comparison)

**Result:** LB = ? (not yet submitted)

---

## Model Architecture (v3 and later)

```
Data → Missingness flags → Interaction features → District stats →
Relative features → Compound risk features → DBSCAN isolation flag →
Spatial KMeans groups (fold splitting only) → DROP_COLS cleanup

Per fold (GroupKFold, 5 folds):
  ├─ Per-fold Target Encoding (district, soil_type, landcover, urban_rural)
  ├─ LightGBM (native categoricals)
  ├─ XGBoost (train-only LabelEncoder)
  └─ CatBoost (train-only LabelEncoder)

OOF predictions → Nelder-Mead blend weights → Final ensemble
Test predictions → weighted average across folds → clip [0,1]
```

---

## Key Lessons

1. **OOF scores lie when validation leaks geographically.** Spatial GroupKFold gave the same OOF but a worse LB than v1 — correctly showing v1's OOF was optimistic.

2. **District was masking, not explaining.** District was the #1 feature by importance but was memorising labels, not flood dynamics. Adding district stats + relative features transferred that signal into interpretable numbers.

3. **Compound features matter more than pairwise.** No feature has >0.081 correlation with target individually. Signal only emerges in combinations of rainfall + drainage + elevation + history.

4. **Missing = signal.** Infrastructure missingness is not random — no electricity / no road quality data implies off-grid or unmeasured location with distinct risk profile.

5. **The metric rewards R², not just RMSE.** The custom metric is `(mae+rmse)/2 * (1+penalty)` where penalty = max(0, 1-r²). Tree ensembles compress prediction variance heavily, so calibration to spread predictions is worth exploring.

---

## What's Left to Try

- Submit v4 (MLP) and v5 (calibrated) to get LB scores
- Optuna hyperparameter tuning for CatBoost
- Stacking — use OOF predictions as meta-features for a 2nd-level model
- Neural network (TabNet) — designed for tabular data
- Leave-one-out target encoding — reduces encoding leakage vs simple mean TE
- More aggressive compound features targeting the custom metric's R² term
