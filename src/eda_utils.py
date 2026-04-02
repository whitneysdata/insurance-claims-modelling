import numpy as np
import pandas as pd


def overdispersion_test(series: pd.Series) -> dict:
    """
    Compute dispersion index for a count variable.

    Dispersion index = variance / mean.
    Values > 1 indicate overdispersion — Negative Binomial
    regression is recommended over Poisson.

    Parameters
    ----------
    series : pd.Series — count variable (e.g. ClaimNb)

    Returns
    -------
    dict with mean, variance, dispersion index and verdict
    """
    mean = series.mean()
    var  = series.var()
    disp = var / mean

    result = {
        'mean'      : round(mean, 6),
        'variance'  : round(var,  6),
        'dispersion': round(disp, 4),
        'verdict'   : 'Overdispersed — use Negative Binomial' if disp > 1
                      else 'Not overdispersed — Poisson is appropriate'
    }

    print(f"Overdispersion Test:")
    print(f"   Mean        : {result['mean']}")
    print(f"   Variance    : {result['variance']}")
    print(f"   Dispersion  : {result['dispersion']}")
    print(f"   Verdict     : {result['verdict']}")

    return result


def severity_summary(series: pd.Series) -> pd.DataFrame:
    """
    Print summary statistics for a severity (ClaimAmount) series.

    Parameters
    ----------
    series : pd.Series — claim amounts (positive values only)

    Returns
    -------
    pd.DataFrame — summary statistics
    """
    stats = {
        'Count'   : len(series),
        'Mean'    : series.mean(),
        'Median'  : series.median(),
        'Std'     : series.std(),
        'Min'     : series.min(),
        'P25'     : series.quantile(0.25),
        'P75'     : series.quantile(0.75),
        'P95'     : series.quantile(0.95),
        'P99'     : series.quantile(0.99),
        'Max'     : series.max(),
        'Skewness': series.skew(),
        'Kurtosis': series.kurtosis()
    }

    summary = pd.DataFrame.from_dict(
        stats, orient='index', columns=['Value']
    ).round(4)

    print("Severity Summary Statistics:")
    print(summary.to_string())
    return summary


def get_numeric_cols(df: pd.DataFrame,
                     exclude: list = None) -> list:
    """
    Return numeric column names optionally excluding some.

    Parameters
    ----------
    df      : pd.DataFrame
    exclude : list of column names to exclude

    Returns
    -------
    list of numeric column names
    """
    exclude   = exclude or []
    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in num_cols if c not in exclude]
    