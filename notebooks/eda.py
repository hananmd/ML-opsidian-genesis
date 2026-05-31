"""EDA for flood_risk_score prediction.

Run:
    python notebooks/eda.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "notebooks" / "eda_output"
OUT.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")

for df in (train, test):
    df["nearest_evac_km"] = pd.to_numeric(df["nearest_evac_km"], errors="coerce")

TARGET = "flood_risk_score"
print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")
print(f"\nTarget stats:\n{train[TARGET].describe()}")

# ── 1. Target distribution ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
train[TARGET].hist(bins=50, ax=axes[0], color="steelblue", edgecolor="white")
axes[0].set_title("flood_risk_score distribution")
axes[0].set_xlabel("flood_risk_score")

train[TARGET].plot.kde(ax=axes[1], color="steelblue")
axes[1].set_title("flood_risk_score KDE")
plt.tight_layout()
plt.savefig(OUT / "01_target_distribution.png", dpi=120)
plt.close()
print("Saved 01_target_distribution.png")

# ── 2. Missing values ───────────────────────────────────────────────────────
missing = (train.isnull().mean() * 100).sort_values(ascending=False)
missing = missing[missing > 0]
print(f"\nColumns with missing values (train):\n{missing.to_string()}")

fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.35)))
missing.plot.barh(ax=ax, color="coral")
ax.set_title("Missing value % (train)")
ax.set_xlabel("% missing")
plt.tight_layout()
plt.savefig(OUT / "02_missing_values.png", dpi=120)
plt.close()
print("Saved 02_missing_values.png")

# ── 3. Numeric correlations with target ─────────────────────────────────────
num_cols = train.select_dtypes(include=np.number).columns.tolist()
num_cols = [c for c in num_cols if c != TARGET and c != "record_id"]
corrs = train[num_cols + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()

fig, ax = plt.subplots(figsize=(8, max(4, len(corrs) * 0.3)))
corrs.plot.barh(ax=ax, color=["coral" if v < 0 else "steelblue" for v in corrs])
ax.set_title("Pearson correlation with flood_risk_score")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig(OUT / "03_numeric_correlations.png", dpi=120)
plt.close()
print("Saved 03_numeric_correlations.png")
print(f"\nTop 10 positive correlations:\n{corrs.tail(10).to_string()}")
print(f"\nTop 10 negative correlations:\n{corrs.head(10).to_string()}")

# ── 4. Categorical feature vs target ────────────────────────────────────────
cat_cols = ["district", "soil_type", "landcover", "urban_rural",
            "water_presence_flag", "electricity", "road_quality"]
cat_cols = [c for c in cat_cols if c in train.columns]

fig, axes = plt.subplots(len(cat_cols), 1, figsize=(12, len(cat_cols) * 3.5))
if len(cat_cols) == 1:
    axes = [axes]
for ax, col in zip(axes, cat_cols):
    order = train.groupby(col)[TARGET].median().sort_values().index
    sns.boxplot(data=train, x=col, y=TARGET, order=order, ax=ax,
                palette="muted", fliersize=2)
    ax.set_title(f"{col} vs flood_risk_score")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(OUT / "04_categorical_vs_target.png", dpi=120)
plt.close()
print("Saved 04_categorical_vs_target.png")

# ── 5. Geographic scatter ────────────────────────────────────────────────────
if "latitude" in train.columns and "longitude" in train.columns:
    sample = train.dropna(subset=["latitude", "longitude"]).sample(min(5000, len(train)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(sample["longitude"], sample["latitude"],
                    c=sample[TARGET], cmap="RdYlGn_r", s=8, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="flood_risk_score")
    ax.set_title("Geographic distribution of flood risk")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(OUT / "05_geographic_scatter.png", dpi=120)
    plt.close()
    print("Saved 05_geographic_scatter.png")

# ── 6. District-level mean risk ──────────────────────────────────────────────
if "district" in train.columns:
    district_risk = train.groupby("district")[TARGET].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    print(f"\nDistrict flood risk (mean):\n{district_risk.to_string()}")
    fig, ax = plt.subplots(figsize=(10, max(4, len(district_risk) * 0.4)))
    district_risk["mean"].plot.barh(ax=ax, color="steelblue", xerr=district_risk["std"])
    ax.set_title("Mean flood_risk_score by district")
    ax.set_xlabel("mean flood_risk_score")
    plt.tight_layout()
    plt.savefig(OUT / "06_district_risk.png", dpi=120)
    plt.close()
    print("Saved 06_district_risk.png")

# ── 7. Key scatter plots ─────────────────────────────────────────────────────
scatter_pairs = [
    ("elevation_m", TARGET),
    ("distance_to_river_m", TARGET),
    ("rainfall_7d_mm", TARGET),
    ("historical_flood_count", TARGET),
    ("drainage_index", TARGET),
    ("ndvi", TARGET),
]
scatter_pairs = [(a, b) for a, b in scatter_pairs if a in train.columns]

n = len(scatter_pairs)
fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(14, 8))
axes = axes.flatten()
for ax, (xcol, ycol) in zip(axes, scatter_pairs):
    sample = train[[xcol, ycol]].dropna().sample(min(3000, len(train)), random_state=42)
    ax.scatter(sample[xcol], sample[ycol], s=4, alpha=0.3, color="steelblue")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"{xcol} vs target")
for ax in axes[len(scatter_pairs):]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig(OUT / "07_scatter_plots.png", dpi=120)
plt.close()
print("Saved 07_scatter_plots.png")

# ── 8. Feature importance via RandomForest ───────────────────────────────────
DROP = ["record_id", TARGET, "generation_date", "place_name",
        "reason_not_good_to_live", "is_good_to_live", "is_synthetic",
        "flood_occurrence_current_event", "inundation_area_sqm"]
feat_cols = [c for c in train.columns if c not in DROP]

X_rf = train[feat_cols].copy()
for c in X_rf.select_dtypes(include="object").columns:
    X_rf[c] = X_rf[c].astype("category").cat.codes
X_rf = X_rf.fillna(-999)

rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_rf, train[TARGET])
importances = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
print(f"\nTop 20 feature importances (RF):\n{importances.head(20).to_string()}")

fig, ax = plt.subplots(figsize=(10, 8))
importances.head(25).sort_values().plot.barh(ax=ax, color="steelblue")
ax.set_title("Top 25 feature importances (RandomForest)")
plt.tight_layout()
plt.savefig(OUT / "08_feature_importance.png", dpi=120)
plt.close()
print("Saved 08_feature_importance.png")

# ── 9. Target encoding preview ───────────────────────────────────────────────
for col in ["district", "soil_type", "landcover"]:
    if col in train.columns:
        te = train.groupby(col)[TARGET].mean().sort_values(ascending=False)
        print(f"\nTarget encoding — {col}:\n{te.to_string()}")

print(f"\nAll plots saved to: {OUT}")
print("EDA complete.")
