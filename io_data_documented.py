"""
io_data.py
==========

Input data loading and preprocessing for BESS simulation and penalty analysis.

Purpose
-------
This module handles all file I/O and data preparation tasks: reading the
source Excel file, normalising column names, converting data types,
extracting scalar metadata, and returning a clean time series ready for
downstream simulation and penalty calculations.

It is intentionally decoupled from physical simulation logic, economic
penalty calculations, and optimisation algorithms.

Main Features
-------------
- Excel ingestion with configurable sheet name
- Column renaming from human-readable headers to snake_case identifiers
- Automatic removal of empty and unnamed columns
- Type coercion for datetime and numeric columns
- Extraction of scalar metadata (tariff, factors, KPIs) into a dict
- Time series preparation: sorting, deduplication, and index reset
- Missing-value audit stored in the returned meta dict

Module Structure
----------------
1.  Configuration constants          (DEFAULT_SHEET_NAME, COLUMN_RENAME_MAP,
                                      CORE_COLUMNS, META_COLUMNS)
2.  Internal helpers                 (_normalize_columns, _convert_types)
3.  Public utility functions         (extract_meta, validate_required_columns,
                                      check_missing_values, prepare_timeseries)
4.  Main loader                      (load_input_data)

Data Model
----------
load_input_data() returns two objects:

df_main : pd.DataFrame
    Hourly time series indexed by integer position.  Contains the
    columns listed in CORE_COLUMNS that were present in the source file,
    always starting with ``datetime``, ``forecast``, and ``actual``.

meta : dict
    Scalar parameters and summary KPIs extracted from the first non-null
    value in each META_COLUMNS column, plus a ``missing_values_main_table``
    entry with per-column null counts from df_main.

Scope — Intentional Exclusions
-------------------------------
The following are deliberately NOT implemented here:

- Physical battery simulation
- Balancing penalty calculations
- Optimisation algorithms
- Forecast generation or evaluation

These belong in separate modules so that this I/O layer remains
stable and independently testable.

Author
------
Timur Amirgaliyev

Last Updated
------------
2026-06-10
"""

import pandas as pd
from typing import Tuple, Dict, Any


# =========================================================
# MODULE STRUCTURE
# =========================================================
#
# 1.  Configuration constants      (sheet name, column maps, column lists)
# 2.  Internal helpers             (_normalize_columns, _convert_types)
# 3.  Public utility functions     (extract_meta, validate_required_columns,
#                                   check_missing_values, prepare_timeseries)
# 4.  Main loader                  (load_input_data)
#
# =========================================================


# =========================================================
# 1. CONFIGURATION CONSTANTS
# =========================================================

DEFAULT_SHEET_NAME = "Лист1"

# Mapping from raw Excel header strings to snake_case column names.
# Add entries here whenever the source file gains new columns.
COLUMN_RENAME_MAP = {
    "Date and time":            "datetime",
    "Forecast":                 "forecast",
    "Actual":                   "actual",
    "Deviation":                "deviation",
    "Deviation percentage":     "deviation_pct",
    "Sales forecast":           "sales_forecast",
    "Sales within 5%":          "sales_within_5pct",
    "Sales beyond 5%":          "sales_beyond_5pct",
    "Purchase within 5%":       "purchase_within_5pct",
    "Purchase beyond 5%":       "purchase_beyond_5pct",
    "Total Sales Penalized":    "total_sales_penalized",
    "Unpenalized Sales":        "unpenalized_sales",
    "Loss":                     "loss",
    "ABS Deviation":            "abs_deviation",
    "ABS Deviation percentage": "abs_deviation_pct",
    "Forecast + Actual":        "forecast_plus_actual",
    "(Actual - Forecast)^2":    "sq_error",
    "Tariff":                   "tariff",
    "Increasing factor":        "increasing_factor",
    "Decreasing factor":        "decreasing_factor",
    "Acceptable range (+)":     "acceptable_range_plus",
    "Acceptable range (-)":     "acceptable_range_minus",
    "P installed AC, kW":       "p_installed_ac_kw",
    "MAPE (forecast based)":    "mape_forecast_based",
    "MAE":                      "mae",
    "NMAE":                     "nmae",
    "RMSE":                     "rmse",
    "nRMSE":                    "nrmse",
}

# Columns that form the main hourly time series returned as df_main.
# Only columns present in the source file are kept; the rest are silently skipped.
CORE_COLUMNS = [
    "datetime",
    "forecast",
    "actual",
    "deviation",
    "deviation_pct",
    "sales_forecast",
    "sales_within_5pct",
    "sales_beyond_5pct",
    "purchase_within_5pct",
    "purchase_beyond_5pct",
    "total_sales_penalized",
    "unpenalized_sales",
    "loss",
    "abs_deviation",
    "abs_deviation_pct",
    "forecast_plus_actual",
    "sq_error",
]

# Columns treated as scalar metadata / constants.
# The first non-null value from each column is extracted into the meta dict.
META_COLUMNS = [
    "tariff",
    "increasing_factor",
    "decreasing_factor",
    "acceptable_range_plus",
    "acceptable_range_minus",
    "p_installed_ac_kw",
    "mape_forecast_based",
    "mae",
    "nmae",
    "rmse",
    "nrmse",
    "cumulative_sales_penalized",
    "total_loss",
]


# =========================================================
# 2. INTERNAL HELPERS
# =========================================================

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename known columns to snake_case and drop unnamed/empty columns.

    Steps
    -----
    1. Drop any column whose name starts with ``"Unnamed"``
       (artefacts of blank Excel columns parsed by openpyxl).
    2. Rename columns according to COLUMN_RENAME_MAP.
       Columns not present in the map are left unchanged.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as read from Excel.

    Returns
    -------
    pd.DataFrame
        Copy of df with cleaned column names.
    """
    df = df.copy()

    drop_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.rename(columns=COLUMN_RENAME_MAP)

    return df


def _convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce column dtypes: datetime for the timestamp column, numeric for all others.

    Steps
    -----
    1. Parse ``datetime`` with ``errors="coerce"`` so unparseable values
       become NaT rather than raising.
    2. Apply ``pd.to_numeric(..., errors="coerce")`` to every other column
       so non-numeric cells become NaN rather than raising.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after column normalisation.

    Returns
    -------
    pd.DataFrame
        Copy of df with corrected dtypes.
    """
    df = df.copy()

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    for col in df.columns:
        if col != "datetime":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =========================================================
# 3. PUBLIC UTILITY FUNCTIONS
# =========================================================

def extract_meta(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract scalar metadata from the first non-null value in each META_COLUMNS column.

    Many source Excel files store tariff, penalty factors, and summary KPIs
    as constants repeated down a column (or present only in the first row).
    This function collects those values into a flat dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after column normalisation and type conversion.

    Returns
    -------
    dict
        Keys are the entries of META_COLUMNS that exist in df.
        Values are the first non-null scalar from each column,
        or None if the column is entirely null or absent.
    """
    meta = {}

    for col in META_COLUMNS:
        if col in df.columns:
            non_null = df[col].dropna()
            meta[col] = non_null.iloc[0] if not non_null.empty else None

    return meta


def validate_required_columns(
    df: pd.DataFrame,
    required_columns=None
) -> None:
    """
    Raise an error if any required column is absent from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check (typically after normalisation).
    required_columns : list of str, optional
        Columns that must be present.
        Defaults to ``["datetime", "forecast", "actual"]``.

    Raises
    ------
    ValueError
        If one or more required columns are missing, listing all absent names.
    """
    if required_columns is None:
        required_columns = ["datetime", "forecast", "actual"]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns not found in Excel: {missing}")


def check_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """
    Return a per-column count of null values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to audit.

    Returns
    -------
    dict
        Maps column name → number of null (NaN / NaT) entries.
        Columns with zero nulls are included with value 0.
    """
    return df.isna().sum().to_dict()


def prepare_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a clean hourly time series from the normalised DataFrame.

    Steps
    -----
    1. Keep only columns listed in CORE_COLUMNS that are present in df,
       always placing ``datetime`` first.
    2. Drop rows where ``datetime`` is null (NaT).
    3. Sort ascending by ``datetime``.
    4. Remove duplicate timestamps, keeping the first occurrence.
    5. Reset the integer index starting from 0.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after normalisation, type conversion, and validation.

    Returns
    -------
    pd.DataFrame
        Clean time series ready for simulation and penalty calculations.
    """
    df = df.copy()

    existing_core_cols = [col for col in CORE_COLUMNS if col in df.columns]
    keep_cols = ["datetime"] + [col for col in existing_core_cols if col != "datetime"]
    df = df[keep_cols]

    df = df[df["datetime"].notna()].copy()
    df = df.sort_values("datetime")
    df = df.drop_duplicates(subset="datetime", keep="first")
    df = df.reset_index(drop=True)

    return df


# =========================================================
# 4. MAIN LOADER
# =========================================================

def load_input_data(
    path: str,
    sheet_name: str = DEFAULT_SHEET_NAME
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load, clean, and split an Excel input file into a time series and metadata.

    Processing pipeline
    -------------------
    1. Read the specified sheet from the Excel file using openpyxl.
    2. Drop unnamed columns and rename headers via COLUMN_RENAME_MAP.
    3. Coerce dtypes (datetime + numeric).
    4. Validate that ``datetime``, ``forecast``, and ``actual`` are present.
    5. Extract scalar metadata from META_COLUMNS into a dict.
    6. Build the clean hourly time series via prepare_timeseries().
    7. Append a missing-value audit to meta under ``"missing_values_main_table"``.

    Parameters
    ----------
    path : str
        Absolute or relative path to the Excel file.
    sheet_name : str, optional
        Name of the worksheet to read.  Default ``"Лист1"``.

    Returns
    -------
    df_main : pd.DataFrame
        Clean hourly time series containing the CORE_COLUMNS that were
        present in the source file.  Always includes ``datetime``,
        ``forecast``, and ``actual``.
    meta : dict
        Scalar parameters and KPIs extracted from the file, plus
        ``"missing_values_main_table"`` with per-column null counts.

    Raises
    ------
    ValueError
        If ``datetime``, ``forecast``, or ``actual`` are absent after
        column normalisation.
    FileNotFoundError
        If the path does not point to an existing file (raised by pandas).
    """
    df_raw = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")

    df_raw = _normalize_columns(df_raw)
    df_raw = _convert_types(df_raw)

    validate_required_columns(df_raw, required_columns=["datetime", "forecast", "actual"])

    meta = extract_meta(df_raw)

    df_main = prepare_timeseries(df_raw)

    meta["missing_values_main_table"] = check_missing_values(df_main)

    return df_main, meta


# =========================================================
# QUICK MODULE TEST
# =========================================================

if __name__ == "__main__":
    file_path = "import/korem.xlsx"

    df, meta = load_input_data(file_path)

    print("=== HEAD ===")
    print(df.head())

    print("\n=== COLUMNS ===")
    print(df.columns.tolist())

    print("\n=== META ===")
    for k, v in meta.items():
        print(f"{k}: {v}")

    print("\n=== INFO ===")
    print(f"Rows: {len(df)}")
    print(f"Start: {df['datetime'].min()}")
    print(f"End:   {df['datetime'].max()}")