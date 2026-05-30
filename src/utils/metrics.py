from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def custom_metric(y_true, y_pred) -> float:
    """
    Approximates the competition metric:
    - Balanced error: average of MAE + RMSE
    - Scaled by explained variance penalty (lower R² = higher penalty)
    Lower is better.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)

    balanced_error = (mae + rmse) / 2
    ev_penalty     = 1 + max(0, 1 - r2)
    return balanced_error * ev_penalty
