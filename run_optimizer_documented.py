"""
run_optimizer.py
================

Multi-strategy BESS optimisation runner and scenario comparison.

Purpose
-------
This script evaluates and compares four BESS dispatch strategies over a
historical generation time series:

1. **Baseline** — no BESS; raw penalties on the original actual generation.
2. **Greedy controller** — single-step deviation compensation via
   bess_model.greedy_deviation_controller().
3. **Corridor controller** — single-step command targeting the nearest
   acceptable corridor boundary.
4. **Offline DP** — full-horizon dynamic programming with exact physics
   and the exact economics.py loss function at every node.

All strategies share the same BESSParams configuration and the same
economics evaluation pipeline, making results directly comparable.

Pipeline Overview
-----------------
1. Load input data via io_data.load_input_data().
2. Evaluate the no-BESS baseline with economics.calculate_balancing_penalty().
3. Evaluate the greedy and corridor controllers via
   bess_model.simulate_with_controller() + economics pipeline.
4. (Optional) Run offline_dp_optimize() to compute the globally optimal
   action sequence, then replay it through the same physics layer.
5. Build a summary table with absolute and percentage loss reduction
   relative to the baseline.
6. Export per-step hourly DataFrames and the summary table to Excel.

Offline DP Algorithm
--------------------
The dynamic programming solver uses backward induction over the full
horizon T:

    V(t, s) = min_{a ∈ A}  [ stage_cost(t, s, a)  +  V(t+1, s') ]

where:

    s           = (soc, prev_power_kw, rest_remaining_h, rest_reason)
                  quantised onto discrete grids
    a           = p_cmd_kw, drawn from the action grid
    stage_cost  = loss(t, s, a) + degradation_weight · |a| · dt_h
    s'          = next state from apply_bess_action(s, a)

V() is memoised with lru_cache.  After the backward pass, the optimal
action sequence is reconstructed forward using the same value function.

Grid resolution is controlled by:

    DP_SOC_STEP          — SOC grid spacing [–]
    DP_ACTION_STEP_KW    — action grid spacing [kW]
    DP_REST_STEP_H       — rest-timer grid spacing [h]

Finer grids improve solution quality at the cost of exponentially more
states and computation time.

Output File
-----------
A timestamped Excel workbook written to the ``export/`` directory:

- ``Summary``     — one row per strategy with physical and economic KPIs,
                    absolute and percentage loss reduction vs baseline
- ``Base``        — hourly detail for the no-BESS baseline
- ``Greedy``      — hourly detail for the greedy controller
- ``Corridor``    — hourly detail for the corridor controller
- ``Offline_DP``  — hourly detail for the DP solution (if RUN_OFFLINE_DP)

Scope — Intentional Exclusions
-------------------------------
This script contains no physical modelling, penalty calculation logic,
or I/O.  Those concerns belong in their respective modules:

- Physical battery model     → bess_model.py
- Economic penalties         → economics.py
- Input data loading         → io_data.py
- MILP optimisation          → optimizer_milp.py

Author
------
Timur Amirgaliyev

Last Updated
------------
2026-06-10
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
import sys
from typing import Dict, Any, Optional, Callable, List, Tuple

import numpy as np
import pandas as pd

from io_data import load_input_data
from economics import calculate_balancing_penalty, summarize_penalty
from bess_model import (
    BESSParams,
    BESSState,
    make_initial_state,
    apply_bess_action,
    simulate_with_controller,
    summarize_bess_results,
    greedy_deviation_controller,
)


# =========================================================
# MODULE STRUCTURE
# =========================================================
#
# 1.  User settings                (constants, BESSParams, DP grid params)
# 2.  Helper utilities             (_safe_float, _slug, _require_meta,
#                                   _stage_loss_exact, _corridor_command,
#                                   corridor_controller_factory,
#                                   evaluate_base_case,
#                                   evaluate_controller_case)
# 3.  Offline DP                   (_build_action_grid, _build_soc_grid,
#                                   _build_rest_grid, _nearest_index,
#                                   _quantize_state, _decode_state,
#                                   offline_dp_optimize,
#                                   evaluate_offline_dp_case)
# 4.  Main entry point             (main)
#
# =========================================================


# =========================================================
# 1. USER SETTINGS
# =========================================================

INPUT_FILE = "import/korem.xlsx"
SHEET_NAME = "Лист1"

# ------------------------------------------------------------------
# BESS configuration shared across all controller and DP scenarios.
# ------------------------------------------------------------------
BESS_PARAMS = BESSParams(
    energy_capacity_kwh=120_000,
    p_charge_max_kw=30_000,
    p_discharge_max_kw=30_000,
    soc_min=0.05,
    soc_max=0.95,
    soc_initial=0.50,
    eta_charge=0.95,
    eta_discharge=0.95,
    self_discharge_per_hour=0.0,
    max_delta_p_kw_per_h=None,
    min_rest_after_full_charge_h=1.5,
    min_rest_after_full_discharge_h=1.5,
)

# ------------------------------------------------------------------
# Offline DP grid resolution settings.
#
# Finer steps → higher solution quality, exponentially more states.
# For an initial test the defaults below are a reasonable starting point.
# ------------------------------------------------------------------
RUN_OFFLINE_DP             = True
DP_SOC_STEP                = 0.05    # SOC grid spacing [–]
DP_ACTION_STEP_KW          = 1000.0  # action grid spacing [kW]
DP_REST_STEP_H             = 0.5     # rest-timer grid spacing [h]
DP_TERMINAL_SOC_WEIGHT     = 0.0     # weight on |soc_T - soc_initial| in terminal cost
DP_DEGRADATION_WEIGHT_PER_KWH = 0.0 # per-kWh cost added to the stage objective

# Set to an integer to process only the first N rows (useful for quick tests).
# Set to None to process the full input file.
LIMIT_HOURS: Optional[int] = None

# ------------------------------------------------------------------
# Output path
# ------------------------------------------------------------------
timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"export/run_optimizer_results_{timestamp}.xlsx"


# =========================================================
# 2. HELPER UTILITIES
# =========================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float, returning default on failure or NaN.

    Parameters
    ----------
    x : any
        Value to convert.
    default : float, optional
        Fallback returned when x is None, NaN, or unconvertible.

    Returns
    -------
    float
    """
    try:
        if x is None:
            return float(default)
        if isinstance(x, float) and np.isnan(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _slug(prefix: str) -> str:
    """
    Convert a scenario name to a lowercase snake_case column prefix
    with a trailing underscore.

    Parameters
    ----------
    prefix : str
        Human-readable scenario name (e.g. ``"Greedy Controller"``).

    Returns
    -------
    str
        Normalised prefix (e.g. ``"greedy_controller_"``).
    """
    s = str(prefix).strip().lower().replace(" ", "_").replace("-", "_")
    if not s.endswith("_"):
        s += "_"
    return s


def _require_meta(meta: Dict[str, Any]) -> None:
    """
    Raise KeyError if any required economic parameter is absent from meta.

    Parameters
    ----------
    meta : dict
        Metadata dict as returned by io_data.load_input_data().

    Raises
    ------
    KeyError
        Lists all missing keys in the error message.
    """
    required = [
        "tariff",
        "acceptable_range_plus",
        "acceptable_range_minus",
        "decreasing_factor",
        "increasing_factor",
    ]
    missing = [k for k in required if k not in meta or meta[k] is None]
    if missing:
        raise KeyError(f"Required keys missing from meta: {missing}")


def _stage_loss_exact(forecast: float, fact: float, meta: Dict[str, Any]) -> float:
    """
    Compute the per-step balancing market loss for a single time step.

    This replicates the economics.py penalty formula as a scalar function
    to avoid DataFrame overhead inside the DP inner loop.

    Loss formula
    ------------
    loss = unpenalized_sales - total_sales_penalized

    where total_sales_penalized is constructed from the same five
    components as in calculate_balancing_penalty() — see economics.py
    for the full derivation.

    Parameters
    ----------
    forecast : float
        Scheduled generation [kW].
    fact : float
        Actual generation after BESS [kW].
    meta : dict
        Economic parameters (tariff, acceptable ranges, penalty factors).

    Returns
    -------
    float
        Revenue loss for this step [currency].  A positive value means
        the generator lost money due to imbalance penalties.
    """
    tariff                 = _safe_float(meta["tariff"])
    acceptable_range_plus  = _safe_float(meta["acceptable_range_plus"])
    acceptable_range_minus = _safe_float(meta["acceptable_range_minus"])
    decreasing_factor      = _safe_float(meta["decreasing_factor"])
    increasing_factor      = _safe_float(meta["increasing_factor"])

    deviation = fact - forecast

    if forecast > 0:
        deviation_pct = deviation / forecast
    else:
        deviation_pct = 1.0 if fact != 0 else 0.0

    sales_forecast = forecast * tariff

    # Positive deviation (over-generation)
    positive_dev_pct       = max(deviation_pct, 0.0)
    within_5_pct_positive  = min(positive_dev_pct, acceptable_range_plus)

    if forecast > 0:
        sales_within_5pct = within_5_pct_positive * forecast * tariff
    else:
        sales_within_5pct = acceptable_range_plus * fact * tariff

    beyond_5_pct_positive = max(positive_dev_pct - acceptable_range_plus, 0.0)

    if forecast > 0:
        sales_beyond_5pct = beyond_5_pct_positive * forecast * tariff * decreasing_factor
    else:
        sales_beyond_5pct = beyond_5_pct_positive * fact * tariff * decreasing_factor

    # Negative deviation (under-generation)
    negative_dev_pct       = min(deviation_pct, 0.0)
    within_5_pct_negative  = max(negative_dev_pct, acceptable_range_minus)
    purchase_within_5pct   = within_5_pct_negative * forecast * tariff

    beyond_5_pct_negative = min(negative_dev_pct - acceptable_range_minus, 0.0)

    if forecast > 0:
        purchase_beyond_5pct = beyond_5_pct_negative * forecast * tariff * increasing_factor
    else:
        purchase_beyond_5pct = beyond_5_pct_negative * fact * tariff * increasing_factor

    total_sales_penalized = (
        sales_forecast
        + sales_within_5pct
        + sales_beyond_5pct
        + purchase_within_5pct
        + purchase_beyond_5pct
    )

    unpenalized_sales = fact * tariff
    return float(unpenalized_sales - total_sales_penalized)


def _corridor_command(row: pd.Series, meta: Dict[str, Any]) -> float:
    """
    Compute the BESS power command that moves actual generation to the
    nearest acceptable corridor boundary.

    If actual is above the upper bound → charge (negative command).
    If actual is below the lower bound → discharge (positive command).
    If actual is already inside the corridor → idle (zero command).

    Parameters
    ----------
    row : pd.Series
        Must contain ``"actual"`` and ``"forecast"`` fields.
    meta : dict
        Must contain ``"acceptable_range_plus"`` and
        ``"acceptable_range_minus"``.

    Returns
    -------
    float
        Requested BESS power [kW].
    """
    actual   = _safe_float(row["actual"])
    forecast = _safe_float(row["forecast"])
    acc_plus  = _safe_float(meta["acceptable_range_plus"])
    acc_minus = _safe_float(meta["acceptable_range_minus"])

    if forecast > 0:
        lower_ok = forecast * (1.0 + acc_minus)
        upper_ok = forecast * (1.0 + acc_plus)
    else:
        lower_ok = forecast
        upper_ok = forecast

    if actual > upper_ok:
        return upper_ok - actual   # negative → charge
    if actual < lower_ok:
        return lower_ok - actual   # positive → discharge
    return 0.0


def corridor_controller_factory(
    meta: Dict[str, Any],
) -> Callable[[pd.Series, BESSState, BESSParams], float]:
    """
    Return a corridor controller compatible with the bess_model controller interface.

    The returned controller calls _corridor_command() at every step,
    targeting the nearest acceptable corridor boundary rather than full
    deviation compensation (as the greedy controller does).

    Parameters
    ----------
    meta : dict
        Economic parameters passed through to _corridor_command().

    Returns
    -------
    Callable[[pd.Series, BESSState, BESSParams], float]
        Controller function with the standard signature.
    """
    def controller(row: pd.Series, state: BESSState, params: BESSParams) -> float:
        return _corridor_command(row, meta)
    return controller


def evaluate_base_case(
    df: pd.DataFrame,
    meta: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluate the no-BESS baseline: compute penalties on the raw actual generation.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series with ``"actual"`` and ``"forecast"`` columns.
    meta : dict
        Economic parameters.

    Returns
    -------
    df_out : pd.DataFrame
        Input df with ``"base_"``-prefixed penalty columns appended.
    summary : dict
        Flat summary dict with scenario label, physical placeholders,
        and aggregated economic metrics.
    """
    prefix = "base_"
    df_out = calculate_balancing_penalty(
        df=df.copy(),
        meta=meta,
        actual_col="actual",
        forecast_col="forecast",
        prefix=prefix,
    )
    econ = summarize_penalty(df_out, prefix=prefix)

    summary = {
        "scenario":             "no_bess",
        "optimizer_type":       "base",
        "energy_capacity_kwh":  0.0,
        "p_charge_max_kw":      0.0,
        "p_discharge_max_kw":   0.0,
        "total_loss":           float(econ[f"{prefix}loss"]),
        **econ,
    }
    return df_out, summary


def evaluate_controller_case(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    params: BESSParams,
    controller: Callable[[pd.Series, BESSState, BESSParams], float],
    scenario_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run a controller-based BESS simulation and evaluate its economics.

    Steps
    -----
    1. Simulate the BESS using the supplied controller via
       bess_model.simulate_with_controller().
    2. Compute per-step penalties on ``actual_with_bess`` via
       economics.calculate_balancing_penalty().
    3. Aggregate physical and economic metrics into a summary dict.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series with ``"actual"`` and ``"forecast"`` columns.
    meta : dict
        Economic parameters.
    params : BESSParams
        Physical battery configuration.
    controller : callable
        Strategy function with signature
        ``(row, state, params) -> float``.
    scenario_name : str
        Human-readable label used as the column prefix and summary key.

    Returns
    -------
    df_out : pd.DataFrame
        BESS simulation output with prefixed penalty columns appended.
    summary : dict
        Flat summary dict with physical metrics, economic metrics, and
        scenario metadata.
    """
    prefix = _slug(scenario_name)

    df_bess = simulate_with_controller(
        df=df.copy(),
        controller=controller,
        params=params,
        dt_h=1.0,
        initial_state=None,
        actual_col="actual",
        forecast_col="forecast",
    )

    df_out = calculate_balancing_penalty(
        df=df_bess,
        meta=meta,
        actual_col="actual_with_bess",
        forecast_col="forecast",
        prefix=prefix,
    )

    econ = summarize_penalty(df_out, prefix=prefix)
    bess = summarize_bess_results(df_bess)

    summary = {
        "scenario":             scenario_name,
        "optimizer_type":       "controller",
        **bess,
        "energy_capacity_kwh":  params.energy_capacity_kwh,
        "p_charge_max_kw":      params.p_charge_max_kw,
        "p_discharge_max_kw":   params.p_discharge_max_kw,
        "total_loss":           float(econ[f"{prefix}loss"]),
        **econ,
    }
    return df_out, summary


# =========================================================
# 3. OFFLINE DP
# =========================================================

def _build_action_grid(params: BESSParams, step_kw: float) -> List[float]:
    """
    Build the discrete action grid spanning [-p_charge_max_kw, p_discharge_max_kw].

    Zero is always included.  Values are rounded to 10 decimal places to
    avoid floating-point duplicates after np.arange.

    Parameters
    ----------
    params : BESSParams
        Battery configuration providing charge/discharge limits.
    step_kw : float
        Grid spacing [kW].  Must be > 0.

    Returns
    -------
    list of float
        Sorted list of candidate BESS power commands [kW].

    Raises
    ------
    ValueError
        If step_kw <= 0.
    """
    if step_kw <= 0:
        raise ValueError("DP_ACTION_STEP_KW must be > 0")

    neg = np.arange(-params.p_charge_max_kw, 0.0, step_kw)
    pos = np.arange(0.0, params.p_discharge_max_kw + step_kw, step_kw)

    actions = list(neg) + list(pos)
    actions = sorted(set(float(round(x, 10)) for x in actions))
    if 0.0 not in actions:
        actions.append(0.0)
        actions = sorted(actions)
    return actions


def _build_soc_grid(params: BESSParams, soc_step: float) -> List[float]:
    """
    Build the discrete SOC grid covering [soc_min, soc_max].

    Both endpoints are always included.

    Parameters
    ----------
    params : BESSParams
        Battery configuration providing SOC bounds.
    soc_step : float
        Grid spacing [–].  Must be > 0.

    Returns
    -------
    list of float
        Sorted list of SOC values.

    Raises
    ------
    ValueError
        If soc_step <= 0.
    """
    if soc_step <= 0:
        raise ValueError("DP_SOC_STEP must be > 0")

    vals = np.arange(params.soc_min, params.soc_max + soc_step / 2.0, soc_step)
    vals = np.clip(vals, params.soc_min, params.soc_max)
    vals = sorted(set(float(round(x, 10)) for x in vals))
    if vals[0] != params.soc_min:
        vals.insert(0, float(params.soc_min))
    if vals[-1] != params.soc_max:
        vals.append(float(params.soc_max))
    return vals


def _build_rest_grid(params: BESSParams, rest_step_h: float) -> List[float]:
    """
    Build the discrete rest-timer grid from 0 to the maximum rest duration.

    Zero is always included.

    Parameters
    ----------
    params : BESSParams
        Battery configuration providing rest duration parameters.
    rest_step_h : float
        Grid spacing [h].

    Returns
    -------
    list of float
        Sorted list of rest-timer values [h].
    """
    max_rest = max(
        _safe_float(getattr(params, "min_rest_after_full_charge_h",    0.0)),
        _safe_float(getattr(params, "min_rest_after_full_discharge_h", 0.0)),
    )
    vals = np.arange(0.0, max_rest + rest_step_h / 2.0, rest_step_h)
    vals = sorted(set(float(round(x, 10)) for x in vals))
    if 0.0 not in vals:
        vals.insert(0, 0.0)
    return vals


def _nearest_index(grid: List[float], value: float) -> int:
    """
    Return the index of the grid point nearest to value.

    Parameters
    ----------
    grid : list of float
        Sorted grid values.
    value : float
        Query value.

    Returns
    -------
    int
        Index of the nearest grid point.
    """
    arr = np.asarray(grid, dtype=float)
    return int(np.argmin(np.abs(arr - float(value))))


def _quantize_state(
    state: BESSState,
    soc_grid: List[float],
    power_grid: List[float],
    rest_grid: List[float],
) -> Tuple[int, int, int, int]:
    """
    Map a continuous BESSState to a tuple of discrete grid indices.

    The four-tuple (soc_i, p_i, r_i, reason_i) serves as the hashable
    key for the DP memoisation cache.

    rest_reason encoding:
        0 → "none"
        1 → "after_full_charge"
        2 → "after_full_discharge"

    When rest_remaining_h ≤ 1e-12 the reason is always encoded as 0
    regardless of the stored string, because a zero-rest state is
    functionally identical no matter which rest reason is recorded.

    Parameters
    ----------
    state : BESSState
        Continuous battery state to quantise.
    soc_grid : list of float
        Discrete SOC values.
    power_grid : list of float
        Discrete power values [kW].
    rest_grid : list of float
        Discrete rest-timer values [h].

    Returns
    -------
    tuple of int
        (soc_i, p_i, r_i, reason_i)
    """
    reason_map = {
        "none":                 0,
        "after_full_charge":    1,
        "after_full_discharge": 2,
    }

    soc_i = _nearest_index(soc_grid,   float(state.soc))
    p_i   = _nearest_index(power_grid, float(state.prev_power_kw))
    r_i   = _nearest_index(rest_grid,  max(0.0, float(state.rest_remaining_h)))

    if max(0.0, float(state.rest_remaining_h)) <= 1e-12:
        reason_i = 0
    else:
        reason_i = reason_map.get(str(state.rest_reason), 0)

    return soc_i, p_i, r_i, reason_i


def _decode_state(
    key: Tuple[int, int, int, int],
    soc_grid: List[float],
    power_grid: List[float],
    rest_grid: List[float],
) -> BESSState:
    """
    Reconstruct a BESSState from a tuple of discrete grid indices.

    Inverse of _quantize_state().

    Parameters
    ----------
    key : tuple of int
        (soc_i, p_i, r_i, reason_i) as produced by _quantize_state().
    soc_grid : list of float
        Discrete SOC values.
    power_grid : list of float
        Discrete power values [kW].
    rest_grid : list of float
        Discrete rest-timer values [h].

    Returns
    -------
    BESSState
        Reconstructed battery state with grid-quantised field values.
    """
    reason_rev = {
        0: "none",
        1: "after_full_charge",
        2: "after_full_discharge",
    }
    soc_i, p_i, r_i, reason_i = key
    return BESSState(
        soc=float(soc_grid[soc_i]),
        prev_power_kw=float(power_grid[p_i]),
        rest_remaining_h=float(rest_grid[r_i]),
        rest_reason=reason_rev.get(reason_i, "none"),
    )


def offline_dp_optimize(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    params: BESSParams,
    soc_step: float = 0.05,
    action_step_kw: float = 1000.0,
    rest_step_h: float = 0.5,
    terminal_soc_weight: float = 0.0,
    degradation_weight_per_kwh: float = 0.0,
) -> Tuple[List[float], float]:
    """
    Solve the full-horizon BESS dispatch problem via backward-induction DP.

    The solver has perfect foresight over the entire input horizon.  It
    minimises total balancing market loss, using the exact physics from
    apply_bess_action() and the exact loss formula from _stage_loss_exact().

    Algorithm
    ---------
    Backward pass (memoised recursion):

        V(T, s) = terminal_soc_weight · |soc - soc_initial|

        V(t, s) = min_{a ∈ action_grid} [
                      stage_loss(t, s, a)
                    + degradation_weight · throughput(a)
                    + V(t+1, quantise(next_state(s, a)))
                  ]

    Forward pass (greedy action reconstruction using cached V):

        For t = 0 … T-1:
            a*(t) = argmin_{a} [ stage + V(t+1, quantise(next)) ]
            Advance cur_state with a*(t).

    State space is discretised onto (soc_grid × power_grid × rest_grid ×
    rest_reason) and memoised with lru_cache.

    Parameters
    ----------
    df : pd.DataFrame
        Full input time series.  Must contain ``"actual"`` and
        ``"forecast"`` columns.
    meta : dict
        Economic parameters (tariff, penalty factors, corridor bounds).
    params : BESSParams
        Physical battery configuration.
    soc_step : float, optional
        SOC grid spacing.  Default 0.05.
    action_step_kw : float, optional
        Action grid spacing [kW].  Default 1000.0.
    rest_step_h : float, optional
        Rest-timer grid spacing [h].  Default 0.5.
    terminal_soc_weight : float, optional
        Weight on |soc_T - soc_initial| in the terminal cost.
        Default 0.0 (no terminal penalty).
    degradation_weight_per_kwh : float, optional
        Per-kWh throughput cost added to the stage objective.
        Default 0.0 (disabled).

    Returns
    -------
    optimal_actions_kw : list of float
        Optimal p_cmd_kw for each time step.
    optimal_objective : float
        Minimum achievable total objective value (loss + degradation cost).
    """
    if len(df) == 0:
        return [], 0.0

    actual_arr   = df["actual"].astype(float).to_numpy()
    forecast_arr = df["forecast"].astype(float).to_numpy()
    T = len(df)

    action_grid = _build_action_grid(params, step_kw=action_step_kw)
    soc_grid    = _build_soc_grid(params, soc_step=soc_step)
    power_grid  = list(action_grid)
    rest_grid   = _build_rest_grid(params, rest_step_h=rest_step_h)

    initial_state = make_initial_state(params)
    initial_key   = _quantize_state(initial_state, soc_grid, power_grid, rest_grid)

    # Increase recursion limit to accommodate deep backward-pass calls
    sys.setrecursionlimit(max(20000, T + 2000))

    @lru_cache(maxsize=None)
    def V(t: int, soc_i: int, p_i: int, r_i: int, reason_i: int) -> float:
        """Memoised value function for the backward DP pass."""
        if t >= T:
            terminal_soc = soc_grid[soc_i]
            return float(terminal_soc_weight) * abs(terminal_soc - params.soc_initial)

        state = _decode_state((soc_i, p_i, r_i, reason_i), soc_grid, power_grid, rest_grid)

        best      = np.inf
        actual_t  = float(actual_arr[t])
        forecast_t = float(forecast_arr[t])

        for p_cmd_kw in action_grid:
            result, next_state = apply_bess_action(
                actual_kw=actual_t,
                forecast_kw=forecast_t,
                state=state,
                p_cmd_kw=float(p_cmd_kw),
                params=params,
                dt_h=1.0,
            )

            stage_loss = _stage_loss_exact(
                forecast=forecast_t,
                fact=float(result["actual_with_bess"]),
                meta=meta,
            )

            throughput = (
                _safe_float(result.get("charge_energy_input_kwh",     0.0))
                + _safe_float(result.get("discharge_energy_output_kwh", 0.0))
            )
            stage_obj = stage_loss + float(degradation_weight_per_kwh) * throughput

            next_key = _quantize_state(next_state, soc_grid, power_grid, rest_grid)
            total    = stage_obj + V(t + 1, *next_key)

            if total < best:
                best = total

        return float(best)

    # ------------------------------------------------------------------
    # Forward reconstruction: greedily pick the action that minimises
    # stage_cost + V(t+1, next_state) at each step.
    # ------------------------------------------------------------------
    actions: List[float] = []
    cur_state = make_initial_state(params)

    for t in range(T):
        best_action = 0.0
        best_total  = np.inf

        actual_t   = float(actual_arr[t])
        forecast_t = float(forecast_arr[t])

        for p_cmd_kw in action_grid:
            result, next_state = apply_bess_action(
                actual_kw=actual_t,
                forecast_kw=forecast_t,
                state=cur_state,
                p_cmd_kw=float(p_cmd_kw),
                params=params,
                dt_h=1.0,
            )

            stage_loss = _stage_loss_exact(
                forecast=forecast_t,
                fact=float(result["actual_with_bess"]),
                meta=meta,
            )

            throughput = (
                _safe_float(result.get("charge_energy_input_kwh",     0.0))
                + _safe_float(result.get("discharge_energy_output_kwh", 0.0))
            )
            stage_obj = stage_loss + float(degradation_weight_per_kwh) * throughput

            next_key = _quantize_state(next_state, soc_grid, power_grid, rest_grid)
            total    = stage_obj + V(t + 1, *next_key)

            if total < best_total:
                best_total  = total
                best_action = float(p_cmd_kw)

        _, cur_state = apply_bess_action(
            actual_kw=actual_t,
            forecast_kw=forecast_t,
            state=cur_state,
            p_cmd_kw=best_action,
            params=params,
            dt_h=1.0,
        )
        actions.append(best_action)

    optimal_objective = V(0, *initial_key)
    return actions, float(optimal_objective)


def evaluate_offline_dp_case(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    params: BESSParams,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the offline DP optimiser, replay the solution through the physics
    layer, and evaluate its economics.

    Steps
    -----
    1. Call offline_dp_optimize() to obtain the optimal action sequence.
    2. Replay the actions through simulate_with_controller() to produce
       the physically consistent output DataFrame.
    3. Evaluate per-step penalties via calculate_balancing_penalty().
    4. Aggregate physical and economic metrics into a summary dict.

    The DP grid resolution is read from the module-level constants
    DP_SOC_STEP, DP_ACTION_STEP_KW, DP_REST_STEP_H,
    DP_TERMINAL_SOC_WEIGHT, and DP_DEGRADATION_WEIGHT_PER_KWH.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series.
    meta : dict
        Economic parameters.
    params : BESSParams
        Physical battery configuration.

    Returns
    -------
    df_out : pd.DataFrame
        Simulation output with ``"offline_dp_"``-prefixed penalty columns.
    summary : dict
        Flat summary dict with DP settings, physical metrics, and
        aggregated economic metrics.
    """
    actions_kw, optimal_objective = offline_dp_optimize(
        df=df,
        meta=meta,
        params=params,
        soc_step=DP_SOC_STEP,
        action_step_kw=DP_ACTION_STEP_KW,
        rest_step_h=DP_REST_STEP_H,
        terminal_soc_weight=DP_TERMINAL_SOC_WEIGHT,
        degradation_weight_per_kwh=DP_DEGRADATION_WEIGHT_PER_KWH,
    )

    # Replay actions through the physics layer via a controller wrapper
    action_iter = iter(actions_kw)

    def controller_from_actions(
        row: pd.Series, state: BESSState, params_: BESSParams
    ) -> float:
        return float(next(action_iter))

    df_bess = simulate_with_controller(
        df=df.copy(),
        controller=controller_from_actions,
        params=params,
        dt_h=1.0,
        initial_state=None,
        actual_col="actual",
        forecast_col="forecast",
    )

    prefix = "offline_dp_"
    df_out = calculate_balancing_penalty(
        df=df_bess,
        meta=meta,
        actual_col="actual_with_bess",
        forecast_col="forecast",
        prefix=prefix,
    )

    econ = summarize_penalty(df_out, prefix=prefix)
    bess = summarize_bess_results(df_bess)

    summary = {
        "scenario":             "offline_dp",
        "optimizer_type":       "offline_dp",
        **bess,
        "energy_capacity_kwh":  params.energy_capacity_kwh,
        "p_charge_max_kw":      params.p_charge_max_kw,
        "p_discharge_max_kw":   params.p_discharge_max_kw,
        "dp_objective":         float(optimal_objective),
        "dp_soc_step":          DP_SOC_STEP,
        "dp_action_step_kw":    DP_ACTION_STEP_KW,
        "dp_rest_step_h":       DP_REST_STEP_H,
        "total_loss":           float(econ[f"{prefix}loss"]),
        **econ,
    }

    return df_out, summary


# =========================================================
# 4. MAIN ENTRY POINT
# =========================================================

def main() -> None:
    """
    Execute the full multi-strategy comparison pipeline and export results.

    Steps
    -----
    1. Load and optionally truncate the input time series.
    2. Evaluate the no-BESS baseline.
    3. Evaluate the greedy controller.
    4. Evaluate the corridor controller.
    5. (Optional) Evaluate the offline DP strategy.
    6. Build the summary table with absolute and percentage loss
       reduction relative to the baseline.
    7. Export all hourly DataFrames and the summary to Excel.
    """
    df, meta = load_input_data(INPUT_FILE, sheet_name=SHEET_NAME)
    _require_meta(meta)

    if LIMIT_HOURS is not None:
        df = df.head(int(LIMIT_HOURS)).copy()

    # ------------------------------------------------------------------
    # Baseline (no BESS)
    # ------------------------------------------------------------------
    df_base, summary_base = evaluate_base_case(df, meta)

    # ------------------------------------------------------------------
    # Greedy controller
    # ------------------------------------------------------------------
    df_greedy, summary_greedy = evaluate_controller_case(
        df=df,
        meta=meta,
        params=BESS_PARAMS,
        controller=greedy_deviation_controller,
        scenario_name="greedy",
    )

    # ------------------------------------------------------------------
    # Corridor controller
    # ------------------------------------------------------------------
    corridor_controller = corridor_controller_factory(meta)
    df_corridor, summary_corridor = evaluate_controller_case(
        df=df,
        meta=meta,
        params=BESS_PARAMS,
        controller=corridor_controller,
        scenario_name="corridor",
    )

    summary_rows = [summary_base, summary_greedy, summary_corridor]
    sheets = {
        "Base":     df_base,
        "Greedy":   df_greedy,
        "Corridor": df_corridor,
    }

    # ------------------------------------------------------------------
    # Offline DP (optional)
    # ------------------------------------------------------------------
    if RUN_OFFLINE_DP:
        df_offline_dp, summary_offline_dp = evaluate_offline_dp_case(
            df=df,
            meta=meta,
            params=BESS_PARAMS,
        )
        summary_rows.append(summary_offline_dp)
        sheets["Offline_DP"] = df_offline_dp

    # ------------------------------------------------------------------
    # Cross-scenario comparison
    #
    # loss_reduction_abs = base_loss - scenario_loss
    # loss_reduction_pct = loss_reduction_abs / base_loss
    #
    # Positive values mean the scenario recovered more revenue than the
    # baseline; negative values mean it performed worse.
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)

    base_loss = float(summary_base["total_loss"])
    summary_df["loss_reduction_abs"] = base_loss - summary_df["total_loss"]
    summary_df["loss_reduction_pct"] = np.where(
        base_loss != 0,
        (base_loss - summary_df["total_loss"]) / base_loss,
        np.nan,
    )

    # ------------------------------------------------------------------
    # Export to Excel
    # ------------------------------------------------------------------
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        for sheet_name, df_sheet in sheets.items():
            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Done. Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()