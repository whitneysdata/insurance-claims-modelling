import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)


def load_ml_models(models_folder: str = '../models') -> dict:
    """
    Load all ML models saved by Chapter 4.

    Parameters
    ----------
    models_folder : str — path to models folder

    Returns
    -------
    dict with keys:
        rf_frequency, xgb_frequency,
        rf_severity,  xgb_severity
    """
    files = {
        'rf_frequency' : 'rf_frequency.pkl',
        'xgb_frequency': 'xgb_frequency.pkl',
        'rf_severity'  : 'rf_severity.pkl',
        'xgb_severity' : 'xgb_severity.pkl'
    }

    models = {}
    for key, filename in files.items():
        path = os.path.join(models_folder, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                f"Run Chapter 4 first."
            )
        models[key] = joblib.load(path)

    print("ML models loaded from models/ folder.")
    for key, filename in files.items():
        print(f"   {key:<20} <- {filename}")

    return models


def get_ml_predictions(models: dict,
                       X_freq_test: pd.DataFrame,
                       X_sev_test:  pd.DataFrame) -> dict:
    """
    Generate predictions from all ML models.

    Parameters
    ----------
    models      : dict — output of load_ml_models()
    X_freq_test : frequency test features
    X_sev_test  : severity test features

    Returns
    -------
    dict with prediction arrays
    """
    rf_freq_proba  = models['rf_frequency'].predict_proba(
        X_freq_test
    )[:, 1]
    xgb_freq_proba = models['xgb_frequency'].predict_proba(
        X_freq_test
    )[:, 1]
    rf_sev_pred    = models['rf_severity'].predict(X_sev_test)
    xgb_sev_pred   = np.exp(
        models['xgb_severity'].predict(X_sev_test)
    )

    return {
        'rf_freq_proba' : rf_freq_proba,
        'xgb_freq_proba': xgb_freq_proba,
        'rf_sev_pred'   : rf_sev_pred,
        'xgb_sev_pred'  : xgb_sev_pred
    }


def evaluate_regression(name: str,
                         y_true: np.ndarray,
                         y_pred: np.ndarray) -> dict:
    """
    Compute RMSE, MAE and R-squared for a regression model.

    Parameters
    ----------
    name   : str — model name for display
    y_true : array-like — actual values
    y_pred : array-like — predicted values

    Returns
    -------
    dict of metrics
    """
    metrics = {
        'Model'    : name,
        'RMSE'     : round(np.sqrt(
                        mean_squared_error(y_true, y_pred)), 2),
        'MAE'      : round(mean_absolute_error(y_true, y_pred), 2),
        'R-squared': round(r2_score(y_true, y_pred), 4)
    }
    print(f"{name}:")
    print(f"   RMSE      : EUR {metrics['RMSE']:,.2f}")
    print(f"   MAE       : EUR {metrics['MAE']:,.2f}")
    print(f"   R-squared : {metrics['R-squared']}")
    return metrics
    