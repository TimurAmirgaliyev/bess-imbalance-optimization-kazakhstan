"""
optimizer_milp.py
=================

Mixed-Integer Linear Programming (MILP) optimizer for BESS dispatch scheduling.

Purpose
-------
This module formulates and solves a MILP problem that finds the globally
optimal BESS charge/discharge schedule over a full historical horizon,
minimising balancing market penalties.  It is intentionally decoupled from
the physical simulation layer (bess_model.py) and the economic penalty
layer (economics.py), which it calls as post-processing steps.

Main Features
-------------
- Full-horizon MILP formulation via scipy.optimize.milp
- Charge/discharge power limits (inverter constraints)
- SOC dynamics with charge and discharge efficiency
- Self-discharge rate
- Binary mode variable preventing simultaneous charge and discharge
- Optional ramp-rate constraint on net BESS power
- Optional hard terminal SOC target
- Optional throughput-based degradation cost in the objective
- Post-solve economics evaluation via calculate_balancing_penalty()

Module Structure
----------------
1.  Helper utilities             (_safe_float, _slug, _require_meta,
                                  _build_corridor_bounds, _build_loss_coeffs)
2.  Variable index mapping       (VarIndex)
3.  MILP solver                  (solve_milp_schedule)
4.  Public pipeline wrapper      (evaluate_milp_scenario)

MILP Formulation
----------------
Decision variables (per time step t = 0 … T-1):

    c_t   >= 0          charge power [kW]
    d_t   >= 0          discharge power [kW]
    y_t   ∈ {0, 1}      mode: 1 = discharge allowed, 0 = charge allowed
    e_t                 internal energy at end of step t [kWh]
    soc_t               state of charge at end of step t [–]
    p_t                 net BESS power = d_t − c_t [kW]
    a_t                 actual generation with BESS = actual_t + p_t [kW]
    over_t  >= 0        generation above upper corridor bound [kW]
    under_t >= 0        generation below lower corridor bound [kW]

Objective (minimise):

    Σ_t  [ k_over_t · over_t  +  k_under_t · under_t
           + w_deg · (c_t + d_t) ]

where:

    k_over_t  = tariff · max(0, 1 − decreasing_factor)
    k_under_t = tariff · max(0, increasing_factor − 1)
    w_deg     = degradation_weight_per_kwh  (default 0)

Key constraints:

    (1)  p_t = d_t − c_t
    (2)  a_t = actual_t + p_t
    (3)  e_t = soc_t · E_cap
    (4)  Energy balance:
             e_t = (1 − self_discharge) · e_{t-1}
                   + η_charge · c_t
                   − d_t / η_discharge
    (5)  No simultaneous charge/discharge (big-M via binary y_t):
             c_t ≤ P_charge_max · (1 − y_t)
             d_t ≤ P_discharge_max · y_t
    (6)  Ramp-rate (optional):
             |p_t − p_{t-1}| ≤ max_delta_p_kw_per_h
    (7)  Corridor slack definitions:
             over_t  ≥  a_t − upper_ok_t
             under_t ≥  lower_ok_t − a_t
    (8)  Optional hard terminal SOC:
             soc_{T-1} = terminal_soc_target

Corridor bounds:

    upper_ok_t = forecast_t · (1 + acceptable_range_plus)
    lower_ok_t = forecast_t · (1 + acceptable_range_minus)

Known Limitations
-----------------
- Mandatory rest after full charge / discharge (from bess_model.py) is NOT
  modelled.  Exact linearisation of that constraint would require additional
  binary variables and significantly increases model complexity.
- The soft terminal-SOC penalty (terminal_soc_weight) is accepted as a
  parameter but not yet implemented; only the hard equality version is
  supported when terminal_soc_target is set.

Scope — Intentional Exclusions
-------------------------------
The following are deliberately NOT implemented here:

- Physical step-by-step battery simulation  (see bess_model.py)
- Balancing penalty calculation             (see economics.py)
- Input data loading                        (see io_data.py)
- Receding-horizon or rolling MPC logic

Author
------
Timur Amirgaliyev

Last Updated
------------
2026-06-10
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from bess_model import BESSParams
from economics import calculate_balancing_penalty, summarize_penalty

from datetime import datetime

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
except ImportError as e:
    raise ImportError(
        "optimizer_milp.py requires scipy with scipy.optimize.milp.\n"
        "Install scipy and try again."
    ) from e


# =========================================================
# MODULE STRUCTURE
# =========================================================
#
# 1.  Helper utilities         (_safe_float, _slug, _require_meta,
#                               _build_corridor_bounds, _build_loss_coeffs)
# 2.  Variable index mapping   (VarIndex)
# 3.  MILP solver              (solve_milp_schedule)
# 4.  Public pipeline wrapper  (evaluate_milp_scenario)
#
# =========================================================


# =========================================================
# 1. HELPER UTILITIES
# =========================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float, returning default on failure or NaN.

    Parameters
    ----------
    x : any
        Value to convert.
    default : float, optional
        Fallback value returned when x is None, NaN, or unconvertible.

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


def _slug(name: str) -> str:
    """
    Convert a scenario name to a lowercase snake_case column prefix.

    A trailing underscore is always appended so the prefix can be
    concatenated directly with a column name.

    Parameters
    ----------
    name : str
        Human-readable scenario name (e.g. ``"MILP Opt"``).

    Returns
    -------
    str
        Normalised prefix (e.g. ``"milp_opt_"``).
    """
    s = str(name).strip().lower().replace(" ", "_").replace("-", "_")
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


def _build_corridor_bounds(
    forecast: np.ndarray,
    meta: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the per-step acceptable generation corridor.

    For forecast > 0:

        lower_ok_t = forecast_t · (1 + acceptable_range_minus)
        upper_ok_t = forecast_t · (1 + acceptable_range_plus)

    For forecast <= 0 the corridor collapses to forecast itself, which
    preserves model linearity and avoids pathological behaviour at zero
    scheduled generation.

    Parameters
    ----------
    forecast : np.ndarray, shape (T,)
        Scheduled generation values [kW].
    meta : dict
        Must contain ``acceptable_range_plus`` and ``acceptable_range_minus``.

    Returns
    -------
    lower_ok : np.ndarray, shape (T,)
        Lower bound of the acceptable corridor [kW].
    upper_ok : np.ndarray, shape (T,)
        Upper bound of the acceptable corridor [kW].
    """
    acc_plus  = _safe_float(meta["acceptable_range_plus"])
    acc_minus = _safe_float(meta["acceptable_range_minus"])

    lower_ok = np.where(
        forecast > 0,
        forecast * (1.0 + acc_minus),
        forecast
    )
    upper_ok = np.where(
        forecast > 0,
        forecast * (1.0 + acc_plus),
        forecast
    )

    return lower_ok.astype(float), upper_ok.astype(float)


def _build_loss_coeffs(
    forecast: np.ndarray,
    meta: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-step linear penalty coefficients for the MILP objective.

    The penalty is modelled as:

        loss_t = k_over_t · over_t  +  k_under_t · under_t

    where:

        k_over_t  = tariff · max(0, 1 − decreasing_factor)
        k_under_t = tariff · max(0, increasing_factor − 1)

    These coefficients are exact linear equivalents of the economics.py
    penalty formula when forecast > 0.  For forecast <= 0 the same
    coefficient values are used, keeping the model linear and stable.

    Parameters
    ----------
    forecast : np.ndarray, shape (T,)
        Scheduled generation values [kW].  Shape is used only to size
        the output arrays; values are not directly used in the formula.
    meta : dict
        Must contain ``tariff``, ``decreasing_factor``, and
        ``increasing_factor``.

    Returns
    -------
    k_over : np.ndarray, shape (T,)
        Objective coefficient for over-corridor slack variable.
    k_under : np.ndarray, shape (T,)
        Objective coefficient for under-corridor slack variable.
    """
    tariff = _safe_float(meta["tariff"])
    dec    = _safe_float(meta["decreasing_factor"])
    inc    = _safe_float(meta["increasing_factor"])

    k_over  = np.full_like(forecast, fill_value=tariff * max(0.0, 1.0 - dec),  dtype=float)
    k_under = np.full_like(forecast, fill_value=tariff * max(0.0, inc  - 1.0), dtype=float)

    return k_over, k_under


# =========================================================
# 2. VARIABLE INDEX MAPPING
# =========================================================

class VarIndex:
    """
    Map named decision variables to contiguous integer indices in the
    flat solution vector x used by scipy.optimize.milp.

    The total number of variables is 9 · T, laid out as:

        [c_0..c_{T-1} | d_0..d_{T-1} | y_0..y_{T-1} | soc_0..soc_{T-1} |
         e_0..e_{T-1} | p_0..p_{T-1} | a_0..a_{T-1} |
         over_0..over_{T-1} | under_0..under_{T-1}]

    Variable definitions
    --------------------
    c_t     >= 0        charge power at step t [kW]
    d_t     >= 0        discharge power at step t [kW]
    y_t     ∈ {0,1}     mode binary (1 = discharge, 0 = charge)
    soc_t               SOC at end of step t [–]
    e_t                 internal energy at end of step t [kWh]
    p_t                 net BESS power = d_t − c_t [kW]
    a_t                 actual generation with BESS [kW]
    over_t  >= 0        excess above upper corridor [kW]
    under_t >= 0        shortfall below lower corridor [kW]

    Parameters
    ----------
    T : int
        Number of time steps in the optimisation horizon.

    Attributes
    ----------
    n : int
        Total number of decision variables (= 9 · T).
    """

    def __init__(self, T: int):
        self.T = T

        self.c0     = 0
        self.d0     = self.c0     + T
        self.y0     = self.d0     + T
        self.soc0   = self.y0     + T
        self.e0     = self.soc0   + T
        self.p0     = self.e0     + T
        self.a0     = self.p0     + T
        self.over0  = self.a0     + T
        self.under0 = self.over0  + T

        self.n = self.under0 + T

    def c(self, t: int) -> int:
        """Index of charge power variable at step t."""
        return self.c0 + t

    def d(self, t: int) -> int:
        """Index of discharge power variable at step t."""
        return self.d0 + t

    def y(self, t: int) -> int:
        """Index of binary mode variable at step t."""
        return self.y0 + t

    def soc(self, t: int) -> int:
        """Index of SOC variable at step t."""
        return self.soc0 + t

    def e(self, t: int) -> int:
        """Index of energy variable at step t."""
        return self.e0 + t

    def p(self, t: int) -> int:
        """Index of net BESS power variable at step t."""
        return self.p0 + t

    def a(self, t: int) -> int:
        """Index of actual-with-BESS variable at step t."""
        return self.a0 + t

    def over(self, t: int) -> int:
        """Index of over-corridor slack variable at step t."""
        return self.over0 + t

    def under(self, t: int) -> int:
        """Index of under-corridor slack variable at step t."""
        return self.under0 + t


# =========================================================
# 3. MILP SOLVER
# =========================================================

def solve_milp_schedule(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    params: BESSParams,
    *,
    enforce_nonnegative_actual_with_bess: bool = True,
    terminal_soc_target: Optional[float] = None,
    terminal_soc_weight: float = 0.0,
    degradation_weight_per_kwh: float = 0.0,
) -> Dict[str, Any]:
    """
    Formulate and solve the full-horizon BESS MILP.

    The solver finds the charge/discharge schedule that minimises total
    balancing market penalties over all T time steps simultaneously.

    See the module docstring for the complete mathematical formulation.

    Known limitation
    ----------------
    Mandatory rest after full charge / discharge (bess_model.py parameter
    min_rest_after_full_charge_h / min_rest_after_full_discharge_h) is NOT
    enforced here.  The returned schedule may therefore occasionally violate
    that constraint; use bess_model.simulate_with_actions() to evaluate the
    physically realistic outcome after the fact.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series.  Must contain ``"actual"`` and ``"forecast"``
        columns.
    meta : dict
        Economic parameters.  Required keys: ``tariff``,
        ``acceptable_range_plus``, ``acceptable_range_minus``,
        ``decreasing_factor``, ``increasing_factor``.
    params : BESSParams
        Physical battery configuration.
    enforce_nonnegative_actual_with_bess : bool, optional
        If True, add a lower bound of 0 on a_t (actual with BESS cannot
        go negative).  Default True.
    terminal_soc_target : float or None, optional
        If given, adds a hard equality constraint forcing
        soc_{T-1} = terminal_soc_target.  Default None.
    terminal_soc_weight : float, optional
        Reserved for a future soft terminal-SOC penalty term.
        Currently unused.  Default 0.0.
    degradation_weight_per_kwh : float, optional
        Cost coefficient added to c_t and d_t in the objective to proxy
        battery degradation through throughput.  Default 0.0 (disabled).

    Returns
    -------
    dict
        Keys:

        - ``status``           — solver status string from scipy
        - ``success``          — bool, True if an optimal solution was found
        - ``objective_value``  — optimal objective value (total penalty)
        - ``df_solution``      — DataFrame with all decision variable
                                 values and derived columns appended
        - ``raw_result``       — raw scipy OptimizeResult object

    Raises
    ------
    KeyError
        If required columns or meta keys are missing.
    RuntimeError
        If the MILP solver fails to find a feasible solution.
    """
    _require_meta(meta)

    if "actual" not in df.columns or "forecast" not in df.columns:
        raise KeyError("df must contain 'actual' and 'forecast' columns")

    df_local = df.copy().reset_index(drop=True)
    T = len(df_local)

    if T == 0:
        return {
            "status": "empty_input",
            "success": True,
            "objective_value": 0.0,
            "df_solution": df_local.copy(),
            "raw_result": None,
        }

    actual   = df_local["actual"].astype(float).to_numpy()
    forecast = df_local["forecast"].astype(float).to_numpy()

    lower_ok, upper_ok = _build_corridor_bounds(forecast, meta)
    k_over,   k_under  = _build_loss_coeffs(forecast, meta)

    idx = VarIndex(T)

    # ------------------------------------------------------------------
    # Objective vector
    # scipy.optimize.milp minimises c_obj @ x
    # ------------------------------------------------------------------
    c_obj = np.zeros(idx.n, dtype=float)

    # Penalty for corridor violations
    for t in range(T):
        c_obj[idx.over(t)]  = k_over[t]
        c_obj[idx.under(t)] = k_under[t]

    # Optional throughput-based degradation proxy: penalise c_t + d_t
    if degradation_weight_per_kwh != 0.0:
        for t in range(T):
            c_obj[idx.c(t)] += float(degradation_weight_per_kwh)
            c_obj[idx.d(t)] += float(degradation_weight_per_kwh)

    # Soft terminal-SOC penalty: placeholder for future implementation
    if terminal_soc_target is not None and terminal_soc_weight > 0.0:
        pass  # not yet implemented

    # ------------------------------------------------------------------
    # Variable bounds
    # ------------------------------------------------------------------
    lb = np.full(idx.n, -np.inf, dtype=float)
    ub = np.full(idx.n,  np.inf, dtype=float)

    e_min = params.soc_min * params.energy_capacity_kwh
    e_max = params.soc_max * params.energy_capacity_kwh

    for t in range(T):
        lb[idx.c(t)]   = 0.0;                  ub[idx.c(t)]   = params.p_charge_max_kw
        lb[idx.d(t)]   = 0.0;                  ub[idx.d(t)]   = params.p_discharge_max_kw
        lb[idx.y(t)]   = 0.0;                  ub[idx.y(t)]   = 1.0
        lb[idx.soc(t)] = params.soc_min;        ub[idx.soc(t)] = params.soc_max
        lb[idx.e(t)]   = e_min;                 ub[idx.e(t)]   = e_max
        lb[idx.p(t)]   = -params.p_charge_max_kw; ub[idx.p(t)] = params.p_discharge_max_kw
        lb[idx.over(t)]  = 0.0
        lb[idx.under(t)] = 0.0
        if enforce_nonnegative_actual_with_bess:
            lb[idx.a(t)] = 0.0

    bounds = Bounds(lb, ub)

    # Binary integrality for mode variables y_t only
    integrality = np.zeros(idx.n, dtype=int)
    for t in range(T):
        integrality[idx.y(t)] = 1

    # ------------------------------------------------------------------
    # Constraint builder helpers
    # ------------------------------------------------------------------
    rows = []
    bl   = []
    bu   = []

    def add_eq(coeffs: Dict[int, float], rhs: float) -> None:
        """Add equality constraint: coeffs @ x == rhs."""
        row = np.zeros(idx.n, dtype=float)
        for j, val in coeffs.items():
            row[j] = val
        rows.append(row)
        bl.append(rhs)
        bu.append(rhs)

    def add_le(coeffs: Dict[int, float], rhs: float) -> None:
        """Add inequality constraint: coeffs @ x <= rhs."""
        row = np.zeros(idx.n, dtype=float)
        for j, val in coeffs.items():
            row[j] = val
        rows.append(row)
        bl.append(-np.inf)
        bu.append(rhs)

    def add_ge(coeffs: Dict[int, float], rhs: float) -> None:
        """Add inequality constraint: coeffs @ x >= rhs."""
        row = np.zeros(idx.n, dtype=float)
        for j, val in coeffs.items():
            row[j] = val
        rows.append(row)
        bl.append(rhs)
        bu.append(np.inf)

    # ------------------------------------------------------------------
    # Constraint (1): p_t = d_t - c_t
    # ------------------------------------------------------------------
    for t in range(T):
        add_eq({idx.p(t): 1.0, idx.d(t): -1.0, idx.c(t): 1.0}, 0.0)

    # ------------------------------------------------------------------
    # Constraint (2): a_t = actual_t + p_t
    # ------------------------------------------------------------------
    for t in range(T):
        add_eq({idx.a(t): 1.0, idx.p(t): -1.0}, actual[t])

    # ------------------------------------------------------------------
    # Constraint (3): e_t = soc_t * E_cap
    # ------------------------------------------------------------------
    for t in range(T):
        add_eq({idx.e(t): 1.0, idx.soc(t): -params.energy_capacity_kwh}, 0.0)

    # ------------------------------------------------------------------
    # Constraint (4): energy balance
    #
    # e_t = (1 - self_discharge) * e_{t-1}
    #       + η_charge * c_t
    #       - d_t / η_discharge
    #
    # For t = 0, e_{t-1} = soc_initial * E_cap (given initial condition).
    # ------------------------------------------------------------------
    e0   = params.soc_initial * params.energy_capacity_kwh
    sd   = _safe_float(params.self_discharge_per_hour, 0.0)
    keep = 1.0 - sd

    for t in range(T):
        if t == 0:
            add_eq(
                {
                    idx.e(t): 1.0,
                    idx.c(t): -params.eta_charge,
                    idx.d(t):  1.0 / params.eta_discharge,
                },
                keep * e0,
            )
        else:
            add_eq(
                {
                    idx.e(t):     1.0,
                    idx.e(t - 1): -keep,
                    idx.c(t):     -params.eta_charge,
                    idx.d(t):      1.0 / params.eta_discharge,
                },
                0.0,
            )

    # ------------------------------------------------------------------
    # Constraint (5): no simultaneous charge and discharge (big-M)
    #
    # c_t <= P_charge_max * (1 - y_t)   →  c_t + P_charge_max * y_t <= P_charge_max
    # d_t <= P_discharge_max * y_t      →  d_t - P_discharge_max * y_t <= 0
    # ------------------------------------------------------------------
    for t in range(T):
        add_le(
            {idx.c(t): 1.0, idx.y(t):  params.p_charge_max_kw},
            params.p_charge_max_kw,
        )
        add_le(
            {idx.d(t): 1.0, idx.y(t): -params.p_discharge_max_kw},
            0.0,
        )

    # ------------------------------------------------------------------
    # Constraint (6): ramp-rate on p_t (optional)
    #
    # |p_t - p_{t-1}| <= max_delta_p_kw_per_h
    # ------------------------------------------------------------------
    if params.max_delta_p_kw_per_h is not None:
        ramp = float(params.max_delta_p_kw_per_h)
        for t in range(T):
            if t == 0:
                prev_p = 0.0
                add_le({idx.p(t):  1.0}, prev_p + ramp)
                add_ge({idx.p(t):  1.0}, prev_p - ramp)
            else:
                add_le({idx.p(t): 1.0, idx.p(t - 1): -1.0},  ramp)
                add_ge({idx.p(t): 1.0, idx.p(t - 1): -1.0}, -ramp)

    # ------------------------------------------------------------------
    # Constraint (7): corridor violation slacks
    #
    # over_t  >= a_t - upper_ok_t   →  over_t  - a_t >= -upper_ok_t
    # under_t >= lower_ok_t - a_t   →  under_t + a_t >=  lower_ok_t
    # ------------------------------------------------------------------
    for t in range(T):
        add_ge({idx.over(t):  1.0, idx.a(t): -1.0}, -upper_ok[t])
        add_ge({idx.under(t): 1.0, idx.a(t):  1.0},  lower_ok[t])

    # ------------------------------------------------------------------
    # Constraint (8): hard terminal SOC target (optional)
    # ------------------------------------------------------------------
    if terminal_soc_target is not None:
        add_eq({idx.soc(T - 1): 1.0}, float(terminal_soc_target))

    A = np.vstack(rows) if rows else np.zeros((0, idx.n), dtype=float)
    constraints = LinearConstraint(
        A, np.array(bl, dtype=float), np.array(bu, dtype=float)
    )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    result = milp(
        c=c_obj,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"disp": False},
    )

    if not result.success:
        raise RuntimeError(
            f"MILP solver failed. status={result.status}, message={result.message}"
        )

    x = np.asarray(result.x, dtype=float)

    # ------------------------------------------------------------------
    # Build solution DataFrame
    # ------------------------------------------------------------------
    out = df_local.copy()

    out["charge_power_kw"]       = [x[idx.c(t)]     for t in range(T)]
    out["discharge_power_kw"]    = [x[idx.d(t)]     for t in range(T)]
    out["mode_binary"]           = [x[idx.y(t)]     for t in range(T)]
    out["soc_end"]               = [x[idx.soc(t)]   for t in range(T)]
    out["energy_end_kwh"]        = [x[idx.e(t)]     for t in range(T)]
    out["p_bess_kw"]             = [x[idx.p(t)]     for t in range(T)]
    out["p_cmd_kw"]              = out["p_bess_kw"]
    out["actual_with_bess"]      = [x[idx.a(t)]     for t in range(T)]
    out["milp_over_corridor_kw"] = [x[idx.over(t)]  for t in range(T)]
    out["milp_under_corridor_kw"]= [x[idx.under(t)] for t in range(T)]

    # Derive start-of-step SOC and energy from end-of-step values
    out["soc_start"]         = [params.soc_initial] + out["soc_end"].iloc[:-1].tolist()
    out["energy_start_kwh"]  = [params.soc_initial * params.energy_capacity_kwh] + out["energy_end_kwh"].iloc[:-1].tolist()

    out["deviation_before"]     = out["actual"]          - out["forecast"]
    out["deviation_after_bess"] = out["actual_with_bess"] - out["forecast"]

    # Energy throughput columns (dt_h = 1 h assumed)
    out["charge_energy_input_kwh"]     = out["charge_power_kw"]
    out["discharge_energy_output_kwh"] = out["discharge_power_kw"]

    # Compatibility columns matching bess_model.py output schema
    out["bess_was_clipped"]              = False
    out["bess_hit_min_limit"]            = False
    out["bess_hit_max_limit"]            = False
    out["bess_rest_lock_active_start"]   = False
    out["bess_hit_rest_lock"]            = False
    out["bess_rest_remaining_h_start"]   = 0.0
    out["bess_rest_remaining_h_end"]     = 0.0
    out["bess_rest_reason_start"]        = "none"
    out["bess_rest_reason_end"]          = "none"
    out["bess_rest_started_this_step"]   = False
    out["bess_reached_soc_max_this_step"] = np.isclose(out["soc_end"], params.soc_max)
    out["bess_reached_soc_min_this_step"] = np.isclose(out["soc_end"], params.soc_min)

    return {
        "status":           str(result.status),
        "success":          bool(result.success),
        "objective_value":  float(result.fun),
        "df_solution":      out,
        "raw_result":       result,
    }


# =========================================================
# 4. PUBLIC PIPELINE WRAPPER
# =========================================================

def evaluate_milp_scenario(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    params: BESSParams,
    scenario_name: str = "milp_opt",
    *,
    enforce_nonnegative_actual_with_bess: bool = True,
    terminal_soc_target: Optional[float] = None,
    terminal_soc_weight: float = 0.0,
    degradation_weight_per_kwh: float = 0.0,
) -> Dict[str, Any]:
    """
    Full pipeline: solve MILP, evaluate economics, and return a unified result.

    Steps
    -----
    1. Call solve_milp_schedule() to obtain the optimal power schedule.
    2. Call calculate_balancing_penalty() on ``actual_with_bess`` to
       compute per-step economic metrics.
    3. Call summarize_penalty() for aggregate economic totals.
    4. Combine MILP diagnostics and economic summary into a flat dict.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series with ``"actual"`` and ``"forecast"`` columns.
    meta : dict
        Economic parameters (see solve_milp_schedule for required keys).
    params : BESSParams
        Physical battery configuration.
    scenario_name : str, optional
        Human-readable label for this scenario.  Used as the column prefix
        in the economics output (slugified).  Default ``"milp_opt"``.
    enforce_nonnegative_actual_with_bess : bool, optional
        Passed through to solve_milp_schedule.  Default True.
    terminal_soc_target : float or None, optional
        Passed through to solve_milp_schedule.  Default None.
    terminal_soc_weight : float, optional
        Passed through to solve_milp_schedule.  Default 0.0.
    degradation_weight_per_kwh : float, optional
        Passed through to solve_milp_schedule.  Default 0.0.

    Returns
    -------
    dict
        Keys:

        - ``scenario``    — scenario name string
        - ``hourly_df``   — per-step DataFrame with MILP results and
                            economics columns appended
        - ``summary``     — flat dict with MILP diagnostics, BESSParams,
                            and aggregated economic metrics
        - ``raw_result``  — raw scipy OptimizeResult object

    Raises
    ------
    RuntimeError
        If the MILP solver fails (propagated from solve_milp_schedule).
    """
    prefix = _slug(scenario_name)

    solved = solve_milp_schedule(
        df=df,
        meta=meta,
        params=params,
        enforce_nonnegative_actual_with_bess=enforce_nonnegative_actual_with_bess,
        terminal_soc_target=terminal_soc_target,
        terminal_soc_weight=terminal_soc_weight,
        degradation_weight_per_kwh=degradation_weight_per_kwh,
    )

    df_milp = solved["df_solution"]

    df_econ = calculate_balancing_penalty(
        df=df_milp,
        meta=meta,
        actual_col="actual_with_bess",
        forecast_col="forecast",
        prefix=prefix,
    )

    econ_summary = summarize_penalty(df_econ, prefix=prefix)

    summary = {
        "scenario":       scenario_name,
        "optimizer_type": "milp",
        **asdict(params),
        "milp_status":           solved["status"],
        "milp_success":          solved["success"],
        "milp_objective_value":  solved["objective_value"],
        "final_soc": float(df_econ["soc_end"].iloc[-1]) if len(df_econ) > 0 else np.nan,
        "total_charge_energy_input_kwh":     float(df_econ["charge_energy_input_kwh"].sum())     if "charge_energy_input_kwh"     in df_econ.columns else np.nan,
        "total_discharge_energy_output_kwh": float(df_econ["discharge_energy_output_kwh"].sum()) if "discharge_energy_output_kwh" in df_econ.columns else np.nan,
        "max_charge_power_kw":     float(df_econ["charge_power_kw"].max())    if "charge_power_kw"    in df_econ.columns else np.nan,
        "max_discharge_power_kw":  float(df_econ["discharge_power_kw"].max()) if "discharge_power_kw" in df_econ.columns else np.nan,
        "hours_above_corridor_after_milp": float((df_econ["milp_over_corridor_kw"]  > 1e-9).sum()) if "milp_over_corridor_kw"  in df_econ.columns else np.nan,
        "hours_below_corridor_after_milp": float((df_econ["milp_under_corridor_kw"] > 1e-9).sum()) if "milp_under_corridor_kw" in df_econ.columns else np.nan,
        "total_loss": float(econ_summary.get(f"{prefix}loss", np.nan)),
        **econ_summary,
    }

    return {
        "scenario":    scenario_name,
        "hourly_df":   df_econ,
        "summary":     summary,
        "raw_result":  solved["raw_result"],
    }


# =========================================================
# QUICK MODULE TEST
# =========================================================

if __name__ == "__main__":
    from io_data import load_input_data

    INPUT_FILE = "import/korem.xlsx"
    SHEET_NAME = "Лист1"

    df, meta = load_input_data(INPUT_FILE, sheet_name=SHEET_NAME)

    params = BESSParams(
        energy_capacity_kwh=10000,
        p_charge_max_kw=5000,
        p_discharge_max_kw=5000,
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

    res = evaluate_milp_scenario(
        df=df,
        meta=meta,
        params=params,
        scenario_name="milp_opt",
        enforce_nonnegative_actual_with_bess=True,
        terminal_soc_target=None,
        terminal_soc_weight=0.0,
        degradation_weight_per_kwh=0.0,
    )

    print("\n=== MILP SUMMARY ===")
    print(pd.DataFrame([res["summary"]]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILE = f"export/optimizer_milp_results_{timestamp}.xlsx"

    summary_df = pd.DataFrame([res["summary"]])

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        res["hourly_df"].to_excel(writer, sheet_name="MILP", index=False)

    print(f"\nDone. Results saved to: {OUTPUT_FILE}")