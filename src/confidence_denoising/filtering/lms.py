"""
Least Mean Squares (LMS) adaptive filter baseline.
"""

from typing import Union
import numpy as np


def lms_filter(
    noisy: np.ndarray,
    reference: np.ndarray,
    mu: float = 0.001,
) -> np.ndarray:
    """
    Apply a single-tap Least Mean Squares (LMS) adaptive filter.

    Update equation:
        y[n] = w[n] * x[n]
        e[n] = d[n] - y[n]
        w[n+1] = w[n] + 2 * mu * e[n] * x[n]

    Args:
        noisy: Input noisy signal x[n].
        reference: Desired/clean signal d[n].
        mu: Adaptation step size (learning parameter).

    Returns:
        Filtered estimate yhat[n] as a numpy array.
    """
    x = np.asarray(noisy, dtype=np.float64)
    d = np.asarray(reference, dtype=np.float64)
    n = min(len(x), len(d))

    w = 0.0
    yhat = np.zeros(n, dtype=np.float64)
    for i in range(n):
        yhat[i] = w * x[i]
        error = d[i] - yhat[i]
        w += 2.0 * mu * error * x[i]

    return yhat.astype(np.float32)
