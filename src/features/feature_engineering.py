import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

SEED = 42
TARGET = "flood_risk_score"


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
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


def add_geo_clusters(train_df: pd.DataFrame, test_df: pd.DataFrame, n_clusters: int = 25):
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    coords = all_df[["latitude", "longitude"]].fillna(all_df[["latitude", "longitude"]].median())
    km = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    all_df["geo_cluster"] = km.fit_predict(coords).astype(str)
    n = len(train_df)
    return all_df.iloc[:n].copy(), all_df.iloc[n:].copy()


def add_target_encoding(train_df: pd.DataFrame, test_df: pd.DataFrame, y: pd.Series) -> tuple:
    """Fit target encoding on train, apply to both train and test."""
    te_cols = ["district", "soil_type", "landcover", "urban_rural", "geo_cluster"]
    global_mean = y.mean()
    for col in te_cols:
        if col not in train_df.columns:
            continue
        mean_map = train_df.groupby(col)[TARGET].mean() if TARGET in train_df.columns else (
            pd.Series(y.values, index=train_df.index).groupby(train_df[col]).mean()
        )
        train_df[f"{col}_te"] = train_df[col].map(mean_map).fillna(global_mean)
        test_df[f"{col}_te"] = test_df[col].map(mean_map).fillna(global_mean)
    return train_df, test_df


def encode_categoricals_ensemble(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Label-encode all object/category columns, return encoded frames + cat indices."""
    cat_cols = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
    X_train = X_train.copy()
    X_test = X_test.copy()
    for c in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[c].astype(str), X_test[c].astype(str)])
        le.fit(combined)
        X_train[c] = le.transform(X_train[c].astype(str))
        X_test[c] = le.transform(X_test[c].astype(str))
    cat_indices = [X_train.columns.tolist().index(c) for c in cat_cols]
    return X_train, X_test, cat_cols, cat_indices


def build_features(train: pd.DataFrame, test: pd.DataFrame, y: pd.Series) -> tuple:
    """Full feature engineering pipeline. Returns (X_train, X_test, cat_cols, cat_indices)."""
    for df in (train, test):
        df["nearest_evac_km"] = pd.to_numeric(df["nearest_evac_km"], errors="coerce")

    train = add_interaction_features(train)
    test = add_interaction_features(test)

    train, test = add_geo_clusters(train, test)

    train[TARGET] = y.values
    train, test = add_target_encoding(train, test, y)
    train = train.drop(columns=[TARGET])

    from src.utils.config import DROP_COLS
    drop = [c for c in DROP_COLS + ["record_id"] if c in train.columns]
    features = [c for c in train.columns if c not in drop and c in test.columns]

    X_train = train[features].copy()
    X_test = test[features].copy()

    X_train, X_test, cat_cols, cat_indices = encode_categoricals_ensemble(X_train, X_test)
    return X_train, X_test, cat_cols, cat_indices
