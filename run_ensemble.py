"""Entry point for the LightGBM + XGBoost + CatBoost ensemble.

Run:
    python run_ensemble.py

Outputs data/processed/submission_ensemble.csv
"""
import pandas as pd
import dagshub
import mlflow

from src.utils.config import DAGSHUB_USER, DAGSHUB_REPO
from src.features.preprocessing import load_data
from src.features.feature_engineering import build_features
from src.models.train_ensemble import train_ensemble


def main():
    dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment("flood-risk-ensemble")

    train, test = load_data("data/raw/train.csv", "data/raw/test.csv")
    y = train["flood_risk_score"].astype(float)

    X_train, X_test, cat_cols, cat_indices = build_features(train, test, y)

    oof_preds, test_preds = train_ensemble(X_train, y, X_test, cat_cols, cat_indices)

    sub = pd.read_csv("data/raw/sample_submission.csv")
    sub["flood_risk_score"] = test_preds
    sub.to_csv("data/processed/submission_ensemble.csv", index=False)
    print("Submission saved → data/processed/submission_ensemble.csv")


if __name__ == "__main__":
    main()
