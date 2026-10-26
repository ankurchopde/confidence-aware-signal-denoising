"""
Dataset loader and train/test partition utilities for CSV files and real IoT data.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from confidence_denoising.data.windowing import standardize


def load_first_numeric_csv(file_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Safely load the first numeric column of a CSV file.

    Args:
        file_path: Path to CSV file.

    Returns:
        1D float64 numpy array, or None if no numeric columns found or on error.
    """
    try:
        df = pd.read_csv(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return None
        return df[numeric_cols[0]].values.astype(np.float64)
    except Exception:
        return None


def load_sensor_directory(
    directory_path: Union[str, Path],
    min_length: int = 1024,
) -> List[np.ndarray]:
    """
    Scan a directory for CSV files and extract standardized numeric series.

    Args:
        directory_path: Path to sensor dataset folder.
        min_length: Minimum number of samples required to keep series.

    Returns:
        List of standardized 1D numpy arrays.
    """
    dir_p = Path(directory_path)
    sources: List[np.ndarray] = []
    if not dir_p.exists():
        return sources

    for csv_file in dir_p.rglob("*.csv"):
        arr = load_first_numeric_csv(csv_file)
        if arr is not None and len(arr) >= min_length:
            sources.append(standardize(arr))

    return sources


def prepare_train_test_split(
    x_windows: np.ndarray,
    y_clean: np.ndarray,
    y_labels: np.ndarray,
    test_size: float = 0.12,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split normalized window datasets into train and validation sets.

    Returns:
        (X_train, X_val, Y_reg_train, Y_reg_val, Y_cls_train, Y_cls_val)
        where X has shape (N, window_size, 1).
    """
    # Normalize per window
    from confidence_denoising.data.windowing import normalize_windows

    x_norm = normalize_windows(x_windows)
    y_clean_norm = normalize_windows(y_clean)

    # Expand channel dimension for Conv1D: (N, window_size, 1)
    x_expanded = x_norm[..., np.newaxis]
    y_cls_2d = y_labels.reshape(-1, 1)

    return train_test_split(
        x_expanded,
        y_clean_norm,
        y_cls_2d,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
