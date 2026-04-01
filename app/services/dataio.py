from __future__ import annotations

from typing import Any

import pandas as pd


def load_table(path: str) -> pd.DataFrame:
    """
    Load a dataset from CSV or Excel file using pandas.
    
    Args:
        path: File path to the dataset
        
    Returns:
        pandas DataFrame with the dataset
        
    Raises:
        ValueError: If file type is not supported
    """
    p = path.lower()
    
    if p.endswith(".csv"):
        return pd.read_csv(path)
    
    if p.endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
        return pd.read_excel(path, engine="openpyxl")
    
    raise ValueError(f"Unsupported dataset file type: {path}")


def detect_feature_types(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Detect feature types (numeric vs categorical) and extract unique values for categorical features.
    
    Args:
        df: pandas DataFrame
        exclude_cols: List of column names to exclude (e.g., target column)
        
    Returns:
        Tuple of (feature_types dict, feature_options dict)
        - feature_types: {column_name: "numeric" or "categorical"}
        - feature_options: {column_name: [list of unique values]} for categorical only
    """
    if exclude_cols is None:
        exclude_cols = []
    
    feature_types: dict[str, str] = {}
    feature_options: dict[str, list[str]] = {}
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_types[col] = "numeric"
        else:
            # Categorical
            feature_types[col] = "categorical"
            # Get unique values as strings, sorted
            unique_vals = df[col].dropna().unique()
            feature_options[col] = sorted([str(v) for v in unique_vals])
    
    return feature_types, feature_options
