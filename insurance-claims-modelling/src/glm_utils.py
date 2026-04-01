import os
import joblib
import numpy as np
import pandas as pd


def load_glm_models(models_folder: str = '../models') -> dict:
    """
    Load all fitted GLM models saved by Chapter 3.

    Parameters
    ----------
    models_folder : str — path to models folder

    Returns
    -------
    dict with keys:
        poisson, neg_binomial, gamma,
        lognormal, lognormal_sigma2
    """
    files = {
        'poisson'          : 'poisson.pkl',
        'neg_binomial'     : 'neg_binomial.pkl',
        'gamma'            : 'gamma.pkl',
        'lognormal'        : 'lognormal.pkl',
        'lognormal_sigma2' : 'lognormal_sigma2.pkl'
    }

    models = {}
    for key, filename in files.items():
        path = os.path.join(models_folder, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                f"Run Chapter 3 first to fit and save GLM models."
            )
        models[key] = joblib.load(path)

    print("GLM models loaded from models/ folder.")
    return models


def compare_models(models_folder: str = '../models') -> dict:
    """
    Load and display GLM comparison tables saved by Chapter 3.

    Parameters
    ----------
    models_folder : str — path to models folder

    Returns
    -------
    dict with keys: freq_comparison, sev_comparison
    """
    freq_path = os.path.join(models_folder, 'freq_model_comparison.csv')
    sev_path  = os.path.join(models_folder, 'sev_model_comparison.csv')

    freq_comp = pd.read_csv(freq_path)
    sev_comp  = pd.read_csv(sev_path)

    print("Frequency Model Comparison:")
    print(freq_comp.to_string(index=False))
    print()
    print("Severity Model Comparison:")
    print(sev_comp.to_string(index=False))

    return {
        'freq_comparison': freq_comp,
        'sev_comparison' : sev_comp
    }


def lognormal_predict(model, X: pd.DataFrame,
                      sigma2: float) -> np.ndarray:
    """
    Generate bias-corrected predictions from a Lognormal model.

    Back-transforms log-scale predictions using:
    E[Y] = exp(mu + sigma2/2)

    Parameters
    ----------
    model  : fitted OLS model on log(ClaimAmount)
    X      : pd.DataFrame — features for prediction
    sigma2 : float — residual variance from model fit

    Returns
    -------
    np.ndarray — predicted ClaimAmount on original scale
    """
    log_pred = model.predict(X)
    return np.exp(log_pred + sigma2 / 2)
    