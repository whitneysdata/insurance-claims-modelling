import numpy as np
import pandas as pd
import os


def load_raw_data(data_folder: str = None):
    if data_folder is None:
        current = os.getcwd()
        for _ in range(4):
            candidate = os.path.join(current, 'data')
            if os.path.exists(candidate):
                data_folder = candidate
                break
            current = os.path.dirname(current)

    freq = pd.read_csv(
        os.path.join(data_folder, 'freMTPL2freq.csv.zip'),  # changed
        compression='zip'
    )
    sev = pd.read_csv(
        os.path.join(data_folder, 'freMTPL2sev.csv')
    )

    print(f"Frequency loaded : {freq.shape}")
    print(f"Severity loaded  : {sev.shape}")
    return freq, sev


def load_prepared_data(data_folder: str = '../data'):
    """
    Load all prepared datasets saved by Chapter 1.

    Used by Chapters 2-5 to avoid rerunning the pipeline.

    Parameters
    ----------
    data_folder : str — path to data folder

    Returns
    -------
    dict with keys:
        df            — full merged and encoded dataset
        df_severity   — claims-only severity subset
        X_freq_train, X_freq_test  — frequency features
        X_sev_train,  X_sev_test   — severity features
        y_freq_train, y_freq_test  — ClaimNb targets
        y_sev_train,  y_sev_test   — ClaimAmount targets
        exposure_train, exposure_test — Exposure for GLM offset
    """
    print("Loading prepared data from data/ folder...")

    def load(filename):
        return pd.read_csv(
            os.path.join(data_folder, filename),
            compression='gzip'
        )

    def load_series(filename):
        return load(filename).squeeze()

    data = {
        'df'             : load('full_data.csv.gz'),
        'df_severity'    : load('severity_data.csv.gz'),
        'X_freq_train'   : load('X_freq_train.csv.gz'),
        'X_freq_test'    : load('X_freq_test.csv.gz'),
        'X_sev_train'    : load('X_sev_train.csv.gz'),
        'X_sev_test'     : load('X_sev_test.csv.gz'),
        'y_freq_train'   : load_series('y_freq_train.csv.gz'),
        'y_freq_test'    : load_series('y_freq_test.csv.gz'),
        'y_sev_train'    : load_series('y_sev_train.csv.gz'),
        'y_sev_test'     : load_series('y_sev_test.csv.gz'),
        'exposure_train' : load_series('exposure_train.csv.gz'),
        'exposure_test'  : load_series('exposure_test.csv.gz'),
    }

    print("All datasets loaded successfully.")
    print(f"   Full dataset     : {data['df'].shape}")
    print(f"   Severity subset  : {data['df_severity'].shape}")
    print(f"   Freq train/test  : {data['X_freq_train'].shape} / {data['X_freq_test'].shape}")
    print(f"   Sev  train/test  : {data['X_sev_train'].shape} / {data['X_sev_test'].shape}")

    return data
    