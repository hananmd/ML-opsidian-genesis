# Flood Risk Prediction — Approach Log

## Competition
**ML Opsidian Genesis - Initial Round 26**
Task: Predict `flood_risk_score` (0–1) for 5,300 locations in Sri Lanka.
Metric: RMSE (lower is better).

---

## Step 1 — Baseline (LightGBM)
**File:** `src/baseline_lgbm.py`

- 5-fold cross-validation, KFold shuffle
- LightGBM with native categorical handling
- Early stopping at 100 rounds, max 3000 rounds
- Dropped: `record_id`, `generation_date`, `place_name`, `reason_not_good_to_live`
- Clipped predictions to [0, 1]

**Result:** OOF RMSE = **0.23585** | Public LB = **0.38699** (prior run)

---

## Step 2 — EDA
**File:** `notebooks/eda.py` → outputs to `notebooks/eda_output/`

Key findings:
- Target well-distributed (mean 0.478, std 0.239) — no class imbalance issue
- All numeric features have near-zero linear correlation with target → relationships are non-linear
- Top predictors (Random Forest importance): `extreme_weather_index`, `terrain_roughness_index`, `district`, `distance_to_river_m`, `socioeconomic_status_index`, `latitude/longitude`
- Missing values ~4–8% across many columns — LightGBM handles natively
- District matters: Vavuniya (mean 0.538) vs Colombo (mean 0.410) — 13% range
- Soil type and landcover have very low signal (<1% variance in mean target)

Plots generated:
1. Target distribution
2. Missing values by column
3. Numeric correlations with target
4. Categorical features vs target (boxplots)
5. Geographic scatter (lat/lon colored by risk)
6. District-level mean risk
7. Key scatter plots (elevation, distance to river, rainfall, etc.)
8. Random Forest feature importance

---

## Step 3 — Improved Ensemble Model
**File:** `src/improved_model.py`

### Feature Engineering Added
| Feature | Description |
|---|---|
| `rain_x_flood_count` | `rainfall_7d_mm × historical_flood_count` |
| `elevation_x_river` | `elevation_m × (1 / distance_to_river_m)` |
| `rain_ratio` | `rainfall_7d_mm / monthly_rainfall_mm` |
| `risk_density` | `historical_flood_count × population_density` |
| `infra_x_evac` | `infrastructure_score / nearest_evac_km` |
| `ndwi_x_rain` | `ndwi × rainfall_7d_mm` |
| `low_elevation_flag` | Binary: elevation < 10m |
| `close_river_flag` | Binary: distance_to_river < 500m |
| `geo_cluster` | KMeans(25) on lat/lon coordinates |
| `*_te` | Target encoding for district, soil_type, landcover, urban_rural, geo_cluster |

### Models Trained (5-fold CV each)
| Model | OOF RMSE | Notes |
|---|---|---|
| LightGBM | 0.23644 | Stopped early ~30–105 iter |
| XGBoost | 0.23555 | Stopped ~63–76 iter |
| **CatBoost** | **0.23512** | Best single model, 167–376 iter |

### Ensemble
- Optimal blend weights (Nelder-Mead): LightGBM -11%, XGBoost 15%, CatBoost 96%
- Simple average OOF RMSE: 0.23541
- **Optimal blend OOF RMSE: 0.23511**

### Submission
**File:** `submissions/improved_ensemble.csv`
**Public LB Score: 0.38504** ✅ (best so far)

---

## Results Summary

| Submission | OOF RMSE | Public LB |
|---|---|---|
| Baseline LightGBM | 0.23585 | 0.38699 |
| Improved Ensemble | 0.23511 | **0.38504** |

---

## What's Next (not yet tried)
- Optuna hyperparameter tuning for CatBoost
- Stacking (OOF predictions as meta-features)
- Neural network (TabNet / MLP)
- Better missing value imputation strategies
- Leave-one-out target encoding (reduce leakage)
