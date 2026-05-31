import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import mlflow
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from src.utils.config import N_FOLDS
from src.utils.metrics import custom_metric

SEED = 42

LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=127,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_data_in_leaf=30,
    lambda_l1=0.1,
    lambda_l2=0.1,
    verbose=-1,
    seed=SEED,
)

XGB_PARAMS = dict(
    objective="reg:squarederror",
    eval_metric="rmse",
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_estimators=3000,
    early_stopping_rounds=100,
    verbosity=0,
    random_state=SEED,
)

CAT_PARAMS = dict(
    loss_function="RMSE",
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    iterations=3000,
    early_stopping_rounds=100,
    random_seed=SEED,
    verbose=0,
)


def _optimal_weights(oof_lgb, oof_xgb, oof_cat, y):
    def blend_rmse(w):
        w = np.array(w) / np.array(w).sum()
        return np.sqrt(mean_squared_error(y, w[0]*oof_lgb + w[1]*oof_xgb + w[2]*oof_cat))
    res = minimize(blend_rmse, x0=[1/3, 1/3, 1/3], method="Nelder-Mead",
                   options={"maxiter": 1000, "xatol": 1e-6})
    w = np.array(res.x)
    return w / w.sum()


def train_ensemble(X, y, X_test, cat_cols, cat_indices):
    """
    3-model ensemble: LightGBM + XGBoost + CatBoost with optimal blending.
    Returns (oof_preds, test_preds) clipped to [0, 1].
    """
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    preds_lgb = np.zeros(len(X_test))
    preds_xgb = np.zeros(len(X_test))
    preds_cat = np.zeros(len(X_test))

    with mlflow.start_run(run_name="ensemble_lgb_xgb_cat"):
        mlflow.log_param("models", "LightGBM+XGBoost+CatBoost")
        mlflow.log_param("n_folds", N_FOLDS)
        mlflow.log_param("n_features", X.shape[1])

        # LightGBM
        print("Training LightGBM...")
        for fold, (tr, va) in enumerate(kf.split(X), 1):
            Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
            ytr, yva = y.iloc[tr], y.iloc[va]
            for c in cat_cols:
                Xtr[c] = Xtr[c].astype("category")
                Xva[c] = pd.Categorical(Xva[c], categories=Xtr[c].cat.categories)
            dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat_cols)
            dva = lgb.Dataset(Xva, yva, categorical_feature=cat_cols)
            m = lgb.train(LGB_PARAMS, dtr, num_boost_round=3000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
            Xt = X_test.copy()
            for c in cat_cols:
                Xt[c] = pd.Categorical(Xt[c], categories=Xtr[c].cat.categories)
            oof_lgb[va] = m.predict(Xva, num_iteration=m.best_iteration)
            preds_lgb += m.predict(Xt, num_iteration=m.best_iteration) / N_FOLDS
            print(f"  LGB fold {fold}: rmse={np.sqrt(mean_squared_error(yva, oof_lgb[va])):.5f}")

        # XGBoost
        print("Training XGBoost...")
        for fold, (tr, va) in enumerate(kf.split(X), 1):
            m = xgb.XGBRegressor(**XGB_PARAMS)
            m.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])], verbose=False)
            oof_xgb[va] = m.predict(X.iloc[va])
            preds_xgb += m.predict(X_test) / N_FOLDS
            print(f"  XGB fold {fold}: rmse={np.sqrt(mean_squared_error(y.iloc[va], oof_xgb[va])):.5f}")

        # CatBoost
        print("Training CatBoost...")
        for fold, (tr, va) in enumerate(kf.split(X), 1):
            m = cb.CatBoostRegressor(**CAT_PARAMS, cat_features=cat_indices)
            m.fit(X.iloc[tr], y.iloc[tr], eval_set=(X.iloc[va], y.iloc[va]), use_best_model=True)
            oof_cat[va] = m.predict(X.iloc[va])
            preds_cat += m.predict(X_test) / N_FOLDS
            print(f"  CAT fold {fold}: rmse={np.sqrt(mean_squared_error(y.iloc[va], oof_cat[va])):.5f}")

        # Blend
        weights = _optimal_weights(oof_lgb, oof_xgb, oof_cat, y)
        oof_preds = weights[0]*oof_lgb + weights[1]*oof_xgb + weights[2]*oof_cat
        test_preds = weights[0]*preds_lgb + weights[1]*preds_xgb + weights[2]*preds_cat

        oof_rmse = np.sqrt(mean_squared_error(y, oof_preds))
        oof_custom = custom_metric(y, oof_preds)

        mlflow.log_param("blend_w_lgb", round(float(weights[0]), 4))
        mlflow.log_param("blend_w_xgb", round(float(weights[1]), 4))
        mlflow.log_param("blend_w_cat", round(float(weights[2]), 4))
        mlflow.log_metric("oof_rmse", oof_rmse)
        mlflow.log_metric("oof_lgb_rmse", np.sqrt(mean_squared_error(y, oof_lgb)))
        mlflow.log_metric("oof_xgb_rmse", np.sqrt(mean_squared_error(y, oof_xgb)))
        mlflow.log_metric("oof_cat_rmse", np.sqrt(mean_squared_error(y, oof_cat)))
        mlflow.log_metric("oof_custom_metric", oof_custom)

        print(f"\nBlend weights — LGB:{weights[0]:.3f}  XGB:{weights[1]:.3f}  CAT:{weights[2]:.3f}")
        print(f"OOF RMSE: {oof_rmse:.5f} | Custom: {oof_custom:.5f}")

    return np.clip(oof_preds, 0, 1), np.clip(test_preds, 0, 1)
