"""
Normalized Least Mean Squares (NLMS) adaptive filter baseline.
"""

from typing import Union
import numpy as np


def nlms_filter(
    noisy: np.ndarray,
    reference: np.ndarray,
    mu: float = 0.1,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Apply a single-tap Normalized Least Mean Squares (NLMS) adaptive filter.

    Update equation:
        y[n] = w[n] * x[n]
        e[n] = d[n] - y[n]
        w[n+1] = w[n] + (mu / (x[n]^2 + eps)) * e[n] * x[n]

    Args:
        noisy: Input noisy signal x[n].
        reference: Desired/clean signal d[n].
        mu: Normalized step size parameter.
        eps: Regularization term to prevent division by zero in quiet regions.

    Returns:
        Filtered estimate yhat[n] as a numpy array.
    """
    x = np.asarray(noisy, dtype=np.float64)
    d = np.asarray(reference, dtype=np.float64)
    n = min(len(x), len(d))

    w = 0.0
    yhat = np.zeros(n, dtype=np.float64)
    for i in range(n):
        denom = x[i] ** 2 + eps
        yhat[i] = w * x[i]
        error = d[i] - yhat[i]
        w += (mu / denom) * error * x[i]

    return yhat.astype(np.float32)
