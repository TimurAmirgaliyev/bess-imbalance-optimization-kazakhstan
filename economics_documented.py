"""
balancing_penalty.py
====================

Balancing market penalty calculation for electricity generation schedules.

Purpose
-------
This module computes the economic penalties incurred when actual electricity
generation deviates from the forecast (scheduled) value on the balancing
market.  It is intentionally decoupled from the physical BESS simulation
and optimization logic.

The design goal is to provide a stable, independently testable economic
layer that any controller, optimizer, or reporting module can call without
modification.

Main Features
-------------
- Per-step deviation calculation (absolute and percentage)
- Asymmetric tolerance bands (acceptable_range_plus / acceptable_range_minus)
- Separate penalty factors for over-generation and under-generation
- Supports both pre-BESS and post-BESS actual values via the actual_col
  parameter
- Configurable column prefix for side-by-side scenario comparison
- Aggregated summary statistics via summarize_penalty()

Module Structure
----------------
1.  Penalty calculation per time step    (calculate_balancing_penalty)
2.  Aggregated summary                   (summarize_penalty)

Sign Convention and Terminology
--------------------------------
deviation = actual - forecast

Positive deviation  (actual > forecast):
    The generator produced more than scheduled.
    On many balancing markets this is penalised at a reduced rate
    (decreasing_factor < 1.0 applied beyond the tolerance band).

Negative deviation  (actual < forecast):
    The generator produced less than scheduled.
    This is typically penalised at an elevated rate
    (increasing_factor > 1.0 applied beyond the tolerance band).

Penalty Formula
---------------
For each time step the penalised revenue is:

    Total Sales Penalized =
          Sales Forecast
        + Sales Within Tolerance  (positive deviation, within band)
        + Sales Beyond Tolerance  (positive deviation, beyond band)
        + Purchase Within Tolerance  (negative deviation, within band)
        + Purchase Beyond Tolerance  (negative deviation, beyond band)

where:

    Sales Forecast              = forecast × tariff
    Sales Within Tolerance      = min(dev_pct, range_plus) × forecast × tariff
    Sales Beyond Tolerance      = max(dev_pct - range_plus, 0) × forecast × tariff × decreasing_factor
    Purchase Within Tolerance   = max(dev_pct, range_minus) × forecast × tariff
    Purchase Beyond Tolerance   = min(dev_pct - range_minus, 0) × forecast × tariff × increasing_factor

    dev_pct = (actual - forecast) / forecast  [when forecast > 0]

Loss per step:

    loss = actual × tariff − Total Sales Penalized

Scope — Intentional Exclusions
-------------------------------
The following are deliberately NOT implemented here:

- Physical battery simulation
- SOC tracking or power constraints
- Optimization algorithms
- Forecast generation or evaluation

These belong in separate modules so that this economic layer remains
stable and independently testable.

Author
------
Timur Amirgaliyev

Last Updated
------------
2026-06-10
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from io_data import load_input_data


# =========================================================
# MODULE STRUCTURE
# =========================================================
#
# 1.  Per-step penalty calculation     (calculate_balancing_penalty)
# 2.  Aggregated summary               (summarize_penalty)
#
# =========================================================


# =========================================================
# 1. PER-STEP PENALTY CALCULATION
# =========================================================

def calculate_balancing_penalty(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    actual_col: str = "actual",
    forecast_col: str = "forecast",
    prefix: str = "py_",
) -> pd.DataFrame:
    """
    Compute balancing market penalties for every time step in a schedule.

    The function appends calculated columns to a copy of df and returns
    the extended DataFrame.  The original df is not modified.

    Use actual_col to switch between scenarios:

    - ``"actual"``           → baseline scenario without BESS
    - ``"actual_with_bess"`` → scenario with BESS active

    Use prefix to keep columns from multiple scenarios in one DataFrame:

    - ``"base_"``  → without BESS
    - ``"bess_"``  → with BESS
    - ``"s1_"``    → scenario 1, etc.

    Calculation Steps
    -----------------
    For each row t:

    1.  Absolute and percentage deviation:

            deviation[t]     = actual[t] - forecast[t]
            deviation_pct[t] = deviation[t] / forecast[t]  (if forecast > 0)

    2.  Baseline unpenalised revenue:

            sales_forecast[t] = forecast[t] × tariff

    3.  Positive deviation split (actual > forecast):

            within_positive   = min(deviation_pct[t], range_plus)
            beyond_positive   = max(deviation_pct[t] - range_plus, 0)

            sales_within_5pct  = within_positive  × forecast × tariff
            sales_beyond_5pct  = beyond_positive  × forecast × tariff × decreasing_factor

    4.  Negative deviation split (actual < forecast):

            within_negative   = max(deviation_pct[t], range_minus)
            beyond_negative   = min(deviation_pct[t] - range_minus, 0)

            purchase_within_5pct  = within_negative × forecast × tariff
            purchase_beyond_5pct  = beyond_negative × forecast × tariff × increasing_factor

    5.  Total penalised sales:

            total_sales_penalized = sales_forecast
                                  + sales_within_5pct
                                  + sales_beyond_5pct
                                  + purchase_within_5pct
                                  + purchase_beyond_5pct

    6.  Revenue loss due to penalties:

            unpenalized_sales = actual × tariff
            loss              = unpenalized_sales - total_sales_penalized

    Parameters
    ----------
    df : pd.DataFrame
        Input time series.  Must contain at least forecast_col and actual_col.
    meta : dict
        Market and tariff parameters.  Required keys:

        - ``tariff``                  — energy price [currency/kWh]
        - ``acceptable_range_plus``   — upper deviation tolerance as a
                                        fraction (e.g. 0.05 for ±5 %)
        - ``acceptable_range_minus``  — lower deviation tolerance as a
                                        fraction (e.g. -0.05); must be ≤ 0
        - ``decreasing_factor``       — penalty multiplier for over-generation
                                        beyond the band (typically < 1.0)
        - ``increasing_factor``       — penalty multiplier for under-generation
                                        beyond the band (typically > 1.0)
    actual_col : str, optional
        Column to treat as measured/realised generation.  Default ``"actual"``.
    forecast_col : str, optional
        Column to treat as the scheduled generation.  Default ``"forecast"``.
    prefix : str, optional
        Prefix prepended to all output column names.  Default ``"py_"``.

    Returns
    -------
    pd.DataFrame
        Copy of df with the following columns appended (all prefixed):

        - ``deviation``            — absolute deviation [kW]
        - ``deviation_pct``        — relative deviation [–]
        - ``sales_forecast``       — revenue at scheduled volume
        - ``positive_dev_pct``     — positive part of deviation_pct
        - ``within_5_pct_positive``— positive deviation clipped to band
        - ``sales_within_5pct``    — revenue from in-band over-generation
        - ``beyond_5_pct_positive``— positive deviation beyond band
        - ``sales_beyond_5pct``    — penalised revenue from excess over-gen
        - ``negative_dev_pct``     — negative part of deviation_pct
        - ``within_5_pct_negative``— negative deviation clipped to band
        - ``purchase_within_5pct`` — cost from in-band under-generation
        - ``beyond_5_pct_negative``— negative deviation beyond band
        - ``purchase_beyond_5pct`` — penalised cost from excess under-gen
        - ``total_sales_penalized``— net penalised revenue per step
        - ``unpenalized_sales``    — revenue without any penalties
        - ``loss``                 — revenue lost to penalties

    Raises
    ------
    KeyError
        If forecast_col or actual_col are missing from df, or if any
        required key is absent from meta.
    """
    df = df.copy()

    if forecast_col not in df.columns:
        raise KeyError(f"forecast_col='{forecast_col}' not found in DataFrame")
    if actual_col not in df.columns:
        raise KeyError(f"actual_col='{actual_col}' not found in DataFrame")

    required_meta = [
        "tariff",
        "acceptable_range_plus",
        "acceptable_range_minus",
        "decreasing_factor",
        "increasing_factor",
    ]
    missing_meta = [k for k in required_meta if k not in meta or meta[k] is None]
    if missing_meta:
        raise KeyError(f"Required keys missing from meta: {missing_meta}")

    # ------------------------------------------------------------------
    # Short aliases for readability
    # ------------------------------------------------------------------
    fact               = df[actual_col]
    forecast           = df[forecast_col]
    tariff             = meta["tariff"]
    acceptable_range_plus  = meta["acceptable_range_plus"]
    acceptable_range_minus = meta["acceptable_range_minus"]
    decreasing_factor  = meta["decreasing_factor"]
    increasing_factor  = meta["increasing_factor"]

    # ------------------------------------------------------------------
    # Column name constants (prefix applied once here)
    # ------------------------------------------------------------------
    c_deviation              = f"{prefix}deviation"
    c_deviation_pct          = f"{prefix}deviation_pct"
    c_sales_forecast         = f"{prefix}sales_forecast"
    c_positive_dev_pct       = f"{prefix}positive_dev_pct"
    c_within_5_pct_positive  = f"{prefix}within_5_pct_positive"
    c_sales_within_5pct      = f"{prefix}sales_within_5pct"
    c_beyond_5_pct_positive  = f"{prefix}beyond_5_pct_positive"
    c_sales_beyond_5pct      = f"{prefix}sales_beyond_5pct"
    c_negative_dev_pct       = f"{prefix}negative_dev_pct"
    c_within_5_pct_negative  = f"{prefix}within_5_pct_negative"
    c_purchase_within_5pct   = f"{prefix}purchase_within_5pct"
    c_beyond_5_pct_negative  = f"{prefix}beyond_5_pct_negative"
    c_purchase_beyond_5pct   = f"{prefix}purchase_beyond_5pct"
    c_total_sales_penalized  = f"{prefix}total_sales_penalized"
    c_unpenalized_sales      = f"{prefix}unpenalized_sales"
    c_loss                   = f"{prefix}loss"

    # ------------------------------------------------------------------
    # Step 1 — Deviation (absolute and percentage)
    #
    # deviation_pct = (actual - forecast) / forecast   [if forecast > 0]
    #
    # Edge cases:
    #   forecast = 0, actual ≠ 0  → deviation_pct = 1.0 (full imbalance)
    #   forecast = 0, actual = 0  → deviation_pct = 0.0 (no imbalance)
    # ------------------------------------------------------------------
    df[c_deviation] = fact - forecast

    df[c_deviation_pct] = np.where(
        forecast > 0,
        df[c_deviation] / forecast,
        np.where(fact != 0, 1.0, 0.0)
    )

    # ------------------------------------------------------------------
    # Step 2 — Baseline scheduled revenue
    #
    # sales_forecast = forecast × tariff
    # ------------------------------------------------------------------
    df[c_sales_forecast] = forecast * tariff

    # ------------------------------------------------------------------
    # Step 3 — Positive deviation (over-generation: actual > forecast)
    #
    # Revenue from delivering more than scheduled, split into:
    #   (a) within tolerance band  → paid at full tariff
    #   (b) beyond tolerance band  → paid at tariff × decreasing_factor
    # ------------------------------------------------------------------

    # Positive part of the percentage deviation
    df[c_positive_dev_pct] = df[c_deviation_pct].clip(lower=0)

    # (a) In-band portion: clipped at acceptable_range_plus
    df[c_within_5_pct_positive] = df[c_positive_dev_pct].clip(
        upper=acceptable_range_plus
    )

    df[c_sales_within_5pct] = np.where(
        forecast > 0,
        df[c_within_5_pct_positive] * forecast * tariff,
        acceptable_range_plus * fact * tariff
    )

    # (b) Beyond-band portion: excess above the tolerance threshold
    df[c_beyond_5_pct_positive] = (
        df[c_positive_dev_pct] - acceptable_range_plus
    ).clip(lower=0)

    df[c_sales_beyond_5pct] = np.where(
        forecast > 0,
        df[c_beyond_5_pct_positive] * forecast * tariff * decreasing_factor,
        df[c_beyond_5_pct_positive] * fact   * tariff * decreasing_factor
    )

    # ------------------------------------------------------------------
    # Step 4 — Negative deviation (under-generation: actual < forecast)
    #
    # Cost of delivering less than scheduled, split into:
    #   (a) within tolerance band  → cost at full tariff
    #   (b) beyond tolerance band  → cost at tariff × increasing_factor
    # ------------------------------------------------------------------

    # Negative part of the percentage deviation
    df[c_negative_dev_pct] = df[c_deviation_pct].clip(upper=0)

    # (a) In-band portion: clipped at acceptable_range_minus (≤ 0)
    df[c_within_5_pct_negative] = df[c_negative_dev_pct].clip(
        lower=acceptable_range_minus
    )

    df[c_purchase_within_5pct] = df[c_within_5_pct_negative] * forecast * tariff

    # (b) Beyond-band portion: shortfall exceeding the negative tolerance
    df[c_beyond_5_pct_negative] = (
        df[c_negative_dev_pct] - acceptable_range_minus
    ).clip(upper=0)

    df[c_purchase_beyond_5pct] = np.where(
        forecast > 0,
        df[c_beyond_5_pct_negative] * forecast * tariff * increasing_factor,
        df[c_beyond_5_pct_negative] * fact     * tariff * increasing_factor
    )

    # ------------------------------------------------------------------
    # Step 5 — Total penalised revenue
    #
    # total_sales_penalized =
    #     sales_forecast
    #   + sales_within_5pct      (in-band over-generation bonus)
    #   + sales_beyond_5pct      (out-of-band over-generation, reduced)
    #   + purchase_within_5pct   (in-band under-generation cost)
    #   + purchase_beyond_5pct   (out-of-band under-generation, elevated)
    # ------------------------------------------------------------------
    df[c_total_sales_penalized] = (
        df[c_sales_forecast]
        + df[c_sales_within_5pct]
        + df[c_sales_beyond_5pct]
        + df[c_purchase_within_5pct]
        + df[c_purchase_beyond_5pct]
    )

    # ------------------------------------------------------------------
    # Step 6 — Revenue loss due to penalties
    #
    # unpenalized_sales = actual × tariff   (ideal revenue, no imbalance)
    # loss              = unpenalized_sales - total_sales_penalized
    # ------------------------------------------------------------------
    df[c_unpenalized_sales] = fact * tariff
    df[c_loss] = df[c_unpenalized_sales] - df[c_total_sales_penalized]

    return df


# =========================================================
# 2. AGGREGATED SUMMARY
# =========================================================

def summarize_penalty(df: pd.DataFrame, prefix: str = "py_") -> Dict[str, float]:
    """
    Aggregate penalised revenue columns across all time steps.

    This function is a convenience wrapper that sums the columns produced
    by calculate_balancing_penalty().  It is safe to call even if some
    columns are absent — missing columns return NaN rather than raising.

    Parameters
    ----------
    df : pd.DataFrame
        Output of calculate_balancing_penalty() (or a DataFrame that
        contains at least some of the expected prefixed columns).
    prefix : str, optional
        Column prefix used when calculate_balancing_penalty() was called.
        Default ``"py_"``.

    Returns
    -------
    dict
        Keys (all prefixed) and their column sums:

        - ``sales_forecast``       — total scheduled revenue
        - ``sales_within_5pct``    — revenue from in-band over-generation
        - ``sales_beyond_5pct``    — revenue from out-of-band over-generation
        - ``purchase_within_5pct`` — cost of in-band under-generation
        - ``purchase_beyond_5pct`` — cost of out-of-band under-generation
        - ``total_sales_penalized``— net penalised revenue (sum of above)
        - ``unpenalized_sales``    — ideal revenue with zero deviation
        - ``loss``                 — total revenue lost to penalties
    """
    cols = [
        f"{prefix}sales_forecast",
        f"{prefix}sales_within_5pct",
        f"{prefix}sales_beyond_5pct",
        f"{prefix}purchase_within_5pct",
        f"{prefix}purchase_beyond_5pct",
        f"{prefix}total_sales_penalized",
        f"{prefix}unpenalized_sales",
        f"{prefix}loss",
    ]

    return {
        col: float(df[col].sum()) if col in df.columns else np.nan
        for col in cols
    }


# =========================================================
# QUICK MODULE TEST
# =========================================================

if __name__ == "__main__":
    file_path = "import/korem.xlsx"

    df, meta = load_input_data(file_path)

    df_calc = calculate_balancing_penalty(
        df=df,
        meta=meta,
        actual_col="actual",
        forecast_col="forecast",
        prefix="py_",
    )

    summary = summarize_penalty(df_calc, prefix="py_")

    print("tariff =", meta["tariff"])
    for k, v in summary.items():
        print(k, "=", v)