import numpy as np
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, accuracy_score, confusion_matrix,
)
from src.utils.metrics import custom_metric
from src.utils.config import LGBM_PARAMS, N_FOLDS


def train_evaluate(X, y, X_test, groups, feature_cols, cat_cols):
    """
    GroupKFold training loop with MLflow tracking.
    Returns oof_preds and test_preds.
    """
    kf         = GroupKFold(n_splits=N_FOLDS)
    oof_preds  = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    with mlflow.start_run(run_name="lgbm_huber_groupkfold"):

        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_param("n_folds",      N_FOLDS)
        mlflow.log_param("cv_strategy",  "GroupKFold_district")
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_param("cat_cols",     cat_cols)

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y, groups=groups), 1):
            X_tr,  X_val = X.iloc[tr_idx],  X.iloc[val_idx]
            y_tr,  y_val = y.iloc[tr_idx],  y.iloc[val_idx]

            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(200),
                ],
            )

            val_pred = np.clip(model.predict(X_val), 0, 1)
            oof_preds[val_idx]  = val_pred
            test_preds         += np.clip(model.predict(X_test), 0, 1) / N_FOLDS

            cm_score = custom_metric(y_val, val_pred)
            rmse     = mean_squared_error(y_val, val_pred) ** 0.5
            mae      = mean_absolute_error(y_val, val_pred)
            r2       = r2_score(y_val, val_pred)

            mlflow.log_metric(f"fold{fold}_custom", cm_score)
            mlflow.log_metric(f"fold{fold}_rmse",   rmse)
            mlflow.log_metric(f"fold{fold}_mae",    mae)
            mlflow.log_metric(f"fold{fold}_r2",     r2)

            print(f"Fold {fold} | CustomMetric={cm_score:.4f} RMSE={rmse:.4f} MAE={mae:.4f} R²={r2:.4f}")

        # ── OOF Overall ──────────────────────────────────────────────────────
        oof_custom = custom_metric(y, oof_preds)
        oof_rmse   = mean_squared_error(y, oof_preds) ** 0.5
        oof_mae    = mean_absolute_error(y, oof_preds)
        oof_r2     = r2_score(y, oof_preds)

        mlflow.log_metric("oof_custom_metric", oof_custom)
        mlflow.log_metric("oof_rmse",          oof_rmse)
        mlflow.log_metric("oof_mae",           oof_mae)
        mlflow.log_metric("oof_r2",            oof_r2)

        mlflow.lightgbm.log_model(model, artifact_path="lgbm_model_last_fold")
        print(f"\nOOF | CustomMetric={oof_custom:.4f} RMSE={oof_rmse:.4f} MAE={oof_mae:.4f} R²={oof_r2:.4f}")

        # ── Local Model Verification Block ───────────────────────────────────
        print("\n--- Local Model Verification Report ---")
        y_binary   = (y > 0.5).astype(int)
        oof_binary = (oof_preds > 0.5).astype(int)

        acc = accuracy_score(y_binary, oof_binary)
        cm  = confusion_matrix(y_binary, oof_binary)
        cr  = classification_report(y_binary, oof_binary, output_dict=True)

        print(f"Overall Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_binary, oof_binary))
        print("Confusion Matrix:")
        print(cm)

        mlflow.log_metric("oof_binary_accuracy",   acc)
        mlflow.log_metric("oof_precision_flood",   cr["1"]["precision"])
        mlflow.log_metric("oof_recall_flood",      cr["1"]["recall"])
        mlflow.log_metric("oof_f1_flood",          cr["1"]["f1-score"])
        mlflow.log_metric("oof_precision_noflood", cr["0"]["precision"])
        mlflow.log_metric("oof_recall_noflood",    cr["0"]["recall"])
        mlflow.log_metric("oof_f1_noflood",        cr["0"]["f1-score"])
        mlflow.log_metric("cm_tn", int(cm[0][0]))
        mlflow.log_metric("cm_fp", int(cm[0][1]))
        mlflow.log_metric("cm_fn", int(cm[1][0]))
        mlflow.log_metric("cm_tp", int(cm[1][1]))

    return oof_preds, test_preds
