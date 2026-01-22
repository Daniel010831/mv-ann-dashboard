# src/utils.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5 #FIXED
    r2 = r2_score(y_true, y_pred)
    # handle divide by zero for MAPE:
    nonzero = y_true != 0
    if nonzero.sum() > 0:
        mape = (np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])).mean() * 100
    else:
        mape = np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}
