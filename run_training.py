import pandas as pd
import dagshub
import mlflow

from src.utils.config import DAGSHUB_USER, DAGSHUB_REPO, MLFLOW_EXPERIMENT
from src.features.preprocessing import load_data, prepare_data
from src.models.train import train_evaluate

def main():
    # ── DAGShub MLflow Setup ──────────────────────────────────────────────
    dagshub.init(repo_owner=DAGSHUB_USER, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    # ── Load & Prepare ────────────────────────────────────────────────────
    train, test = load_data("data/raw/train.csv", "data/raw/test.csv")
    X, y, X_test, groups, feature_cols, cat_cols = prepare_data(train, test)

    # ── Train ─────────────────────────────────────────────────────────────
    oof_preds, test_preds = train_evaluate(X, y, X_test, groups, feature_cols, cat_cols)

    # ── Submission ────────────────────────────────────────────────────────
    sub = pd.read_csv("data/raw/sample_submission.csv")
    sub["flood_risk_score"] = test_preds
    sub.to_csv("data/processed/submission_lgbm_baseline.csv", index=False)
    print("Submission saved → submission_lgbm_baseline.csv")

if __name__ == "__main__":
    main()