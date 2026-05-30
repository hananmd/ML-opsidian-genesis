import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from src.utils.config import DROP_COLS, TARGET


def load_data(train_path: str, test_path: str):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    return train, test


def get_feature_cols(train: pd.DataFrame) -> list:
    return [c for c in train.columns if c not in DROP_COLS]


def encode_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Label-encode categoricals fitted on train+test combined."""
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    le = LabelEncoder()
    for col in cat_cols:
        combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col]  = le.transform(X_test[col].astype(str))
    return X_train, X_test, cat_cols


def prepare_data(train: pd.DataFrame, test: pd.DataFrame):
    """Full preprocessing pipeline. Returns X, y, X_test, groups, feature_cols."""
    feature_cols = get_feature_cols(train)

    groups = train["district"].astype(str).fillna("Unknown").values

    X      = train[feature_cols].copy()
    y      = train[TARGET].copy()
    X_test = test[feature_cols].copy()

    X, X_test, cat_cols = encode_categoricals(X, X_test)

    return X, y, X_test, groups, feature_cols, cat_cols
