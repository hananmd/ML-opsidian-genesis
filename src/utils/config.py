TARGET = "flood_risk_score"

DROP_COLS = [
    "record_id", "place_name", "generation_date",
    "flood_risk_score",
    "flood_occurrence_current_event",
    "inundation_area_sqm",
    "is_good_to_live",
    "reason_not_good_to_live",
    "is_synthetic"
    
]

LGBM_PARAMS = {
    "objective":          "huber",
    "alpha":              0.1,
    "metric":             "huber",
    "n_estimators":       1000,
    "learning_rate":      0.05,
    "num_leaves":         63,
    "min_child_samples":  20,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "reg_alpha":          0.1,
    "reg_lambda":         0.1,
    "random_state":       42,
    "n_jobs":             -1,
    "verbose":            -1,
}

N_FOLDS           = 5
DAGSHUB_USER      = "hananmd"
DAGSHUB_REPO      = "ML-opsidian-genesis"
MLFLOW_EXPERIMENT = "flood-risk-baseline"
