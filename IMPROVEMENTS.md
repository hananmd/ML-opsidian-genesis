# Improvements to Add to Current Best Model
**Based on findings from parallel experimentation branch**

---

## Context
Current best submission: **0.38504** (LightGBM + XGBoost + CatBoost ensemble)
These are additional improvements identified through separate experimentation
that can be layered on top of the existing pipeline.

---

## 1. Validation Strategy — Switch to Spatial KFold

**What to change:** Replace random KFold with GroupKFold using KMeans spatial clusters.

**Why:** Random KFold allows the model to see nearby locations in both
train and validation. In geographic data this inflates local CV scores
because nearby locations share the same flood patterns.
Spatial KFold forces validation on geographically separated areas,
giving a more honest local score and better generalisation.

**How to implement:**
```python
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold

# After coordinate imputation
kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
spatial_groups = kmeans.fit_predict(train[['latitude', 'longitude']].values)

# Use as groups in CV
kf = GroupKFold(n_splits=5)
for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y, groups=spatial_groups)):
    ...
```

**Important:** Do NOT pass `spatial_groups` as a feature to the model —
only use it for fold splitting. Adding it as a feature causes leakage.

---

## 2. District-Level Statistical Features

**What to add:** Instead of relying on district as a raw category,
extract what district encodes as explicit numeric features.

**Why:** In our experiments, district was the #1 most important feature
by a large margin. This means the model was memorising district labels
rather than learning actual flood dynamics. These features transfer
that signal into interpretable numbers.

**Features to add:**
```python
# Compute on train+test combined for full geographic coverage
df['district_flood_history_mean'] = df.groupby('district')['historical_flood_count'].transform('mean')
df['district_rainfall_mean']      = df.groupby('district')['rainfall_7d_mm'].transform('mean')
df['district_rainfall_zscore']    = (
    (df['rainfall_7d_mm'] - df['district_rainfall_mean'])
    / (df.groupby('district')['rainfall_7d_mm'].transform('std') + 0.01)
)
df['district_infra_mean'] = df.groupby('district')['infrastructure_score'].transform('mean')
```

**Observed impact:** District importance dropped from 614 → 101 after adding these.

---

## 3. Missingness as Signal

**What to add:** Binary flags before any null filling.

**Why:** Missing values in infrastructure columns are not random —
a missing `road_quality` likely means no road exists.
A missing `electricity` likely means off-grid.
LightGBM treats all nulls the same way; these flags preserve the distinction.

```python
df['electricity_missing']  = df['electricity'].isnull().astype(int)
df['road_quality_missing'] = df['road_quality'].isnull().astype(int)
df['river_dist_missing']   = df['distance_to_river_m'].isnull().astype(int)
df['drainage_missing']     = df['drainage_index'].isnull().astype(int)
```

**Must be computed before any imputation or filling.**

---

## 4. Compound Risk Features (3-Variable Interactions)

**What to add:** Features that combine 3 variables simultaneously.

**Why:** Every feature has under 0.081 correlation with the target individually.
Signal only exists in combinations. Two-variable ratios (which the current
pipeline already has) are not enough — flood risk requires rainfall AND
drainage AND elevation AND history all together.

```python
# Compound flood risk — rainfall load × drainage failure × low elevation
df['compound_flood_risk'] = (
    (df['rainfall_7d_mm'] * df['catchment_saturation'])
    / (df['elevation_m'].clip(lower=0) + 1)
)

# Historical risk under current conditions
df['historical_risk_amplifier'] = (
    (df['historical_flood_count'].fillna(0) + 1) * df['catchment_saturation']
)

# Evacuation difficulty
df['evacuation_risk'] = df['nearest_evac_km'] / (df['infrastructure_score'] + 0.1)
```

---

## 5. Relative Features (District-Normalised)

**What to add:** Normalise absolute values against district averages.

**Why:** Absolute values lose context.
`elevation = 150m` means different things in different districts.
Relative values tell the model whether a location is
unusually low/high/close compared to its neighbours.

```python
# On train+test combined
district_mean_elev  = df.groupby('district')['elevation_m'].transform('mean')
district_mean_river = df.groupby('district')['distance_to_river_m'].transform('mean')

df['relative_elevation']       = df['elevation_m'] - district_mean_elev
df['relative_river_proximity'] = df['distance_to_river_m'] - district_mean_river
```

---

## 6. DBSCAN Isolation Detection

**What to add:** Flag locations that are geographically isolated.

**Why:** Isolated rural locations have distinct flood dynamics —
no nearby infrastructure, harder to evacuate, different drainage patterns.
KMeans cluster from current pipeline assigns every point to a cluster.
DBSCAN specifically identifies outlier points (cluster = -1).

```python
from sklearn.cluster import DBSCAN

coords      = np.radians(df[['latitude', 'longitude']])
eps_radians = 3.0 / 6371.0  # 3km radius
dbscan      = DBSCAN(eps=eps_radians, min_samples=20,
                     algorithm='ball_tree', metric='haversine')
df['is_isolated_location'] = (
    dbscan.fit_predict(coords) == -1
).astype(int)
```

---

## 7. DROP_COLS — Clean Up Pre-Engineered Columns

**Current risk:** The dataset contains pre-engineered versions of raw columns
(log1p, yeojohnson, qmap variants). If these are in the model they
duplicate information and may encode test distribution into train.

**Confirm these are excluded:**
```python
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
"water_presence_flag",   # redundant with ndwi (r=0.679)
```

---

## 8. Label Encoding Fix (if not using target encoding)

**Current risk:** If LabelEncoder is fitted on train+test combined,
test category distributions leak into training.

**Fix:** Fit encoder on train only, handle unseen test labels safely:
```python
le.fit(X_train[col].astype(str))  # train only
known = set(le.classes_)
X_test[col] = X_test[col].astype(str).apply(
    lambda x: x if x in known else le.classes_[0]
)
```
*Not relevant if target encoding is already being used for all categoricals.*

---

## Expected Combined Impact

| Change | Expected Benefit |
|---|---|
| Spatial KFold | More reliable local CV, better geographic generalisation |
| District statistics | Reduces district over-reliance, adds transferable signal |
| Missingness flags | Better predictions for rural/off-grid locations |
| Compound features | Captures multi-variable flood risk signal |
| Relative features | Normalises absolute values against local context |
| DBSCAN isolation | Distinct handling of isolated locations |
| DROP_COLS cleanup | Removes duplicate/leaking pre-engineered columns |

**Most impactful to implement first:** Spatial KFold + District statistics + Compound features.

