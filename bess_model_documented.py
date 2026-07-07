"""
bess_model.py
=============

Battery Energy Storage System (BESS) physical simulation model.

Purpose
-------
This module implements the physical behavior of a battery energy storage
system (BESS). It is intentionally decoupled from optimization logic,
economic penalty calculations, and forecast evaluation metrics.

The design goal is to provide a stable, well-validated physical layer
that any controller, optimizer, or strategy module can call without
modification.

Main Features
-------------
- Battery state-of-charge (SOC) tracking with configurable min/max bounds
- Charge and discharge power constraints (inverter limits + SOC limits)
- Round-trip efficiency losses for charge and discharge
- Self-discharge rate
- Optional ramp-rate constraints on power change between time steps
- Mandatory rest periods after reaching full charge or full discharge
- Controller-based simulation (rule-based or learned policies)
- Action-vector simulation for offline optimization studies

Module Structure
----------------
1.  Battery parameters and state          (BESSParams, BESSState)
2.  Parameter and state validation        (validate_bess_params, validate_state)
3.  SOC / energy helper functions         (clip_soc, soc_to_energy_kwh, ...)
4.  Power constraint calculation          (get_power_limits_kw)
5.  Command clipping                      (clip_power_command_kw)
6.  SOC and state update                  (update_soc)
7.  Single time-step simulation           (apply_bess_action)
8.  Built-in greedy controller            (greedy_deviation_controller)
9.  Multi-step simulation — action array  (simulate_with_actions)
10. Multi-step simulation — controller    (simulate_with_controller)
11. Summary metrics                       (summarize_bess_results)
12. Fast DP kernel (no validation)        (apply_bess_action_fast)

Sign Convention
---------------
Positive power  (+p_bess_kw)  →  BESS discharges (energy flows to grid/load)
Negative power  (-p_bess_kw)  →  BESS charges    (energy absorbed from grid)

SOC is stored as a fraction in [0.0, 1.0].

Energy Update Equations
-----------------------
Discharge step (p_bess_kw ≥ 0):

    E(t+1) = E(t) - (P_discharge / η_discharge) · Δt

Charge step (p_bess_kw < 0):

    E(t+1) = E(t) + |P_charge| · η_charge · Δt

where:
    E       internal battery energy  [kWh]
    P       battery port power       [kW]
    η       one-way efficiency       [–]
    Δt      time step duration       [h]

Notes on Time-Step Units
------------------------
For dt_h = 1.0 the numerical values of average power [kW] and
energy-per-interval [kWh] are identical, so the variables use the
"_kw" suffix throughout.  If dt_h ≠ 1.0, treat all "_kw" inputs
as average power over the interval.

Scope — Intentional Exclusions
-------------------------------
The following are deliberately NOT implemented here:

- Economic penalty functions
- Forecast accuracy metrics
- Optimization algorithms (LP, DP, RL, etc.)
- Dispatch scheduling logic

These belong in separate modules so that this physical layer remains
stable and independently testable.

Author of the documentation
------
Timur Amirgaliyev

Last Updated
------------
10-06-2026
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Callable, Tuple, List

import numpy as np
import pandas as pd


# =========================================================
# 1. BATTERY PARAMETERS AND STATE
# =========================================================

@dataclass
class BESSParams:
    """
    Immutable configuration parameters for a BESS unit.

    Sign convention
    ---------------
    +p_bess_kw  →  discharge  (energy leaves the battery)
    -p_bess_kw  →  charge     (energy enters the battery)

    SOC is stored as a dimensionless fraction: 0.0 … 1.0.

    Attributes
    ----------
    energy_capacity_kwh : float
        Total nameplate energy capacity [kWh].
    p_charge_max_kw : float
        Maximum charge power (magnitude) [kW].
    p_discharge_max_kw : float
        Maximum discharge power [kW].
    soc_min : float
        Lower SOC bound (depth-of-discharge limit).  Default 0.05.
    soc_max : float
        Upper SOC bound.  Default 0.95.
    soc_initial : float
        SOC at the start of a simulation.  Default 0.50.
    eta_charge : float
        One-way charge efficiency (0, 1].  Default 0.95.
    eta_discharge : float
        One-way discharge efficiency (0, 1].  Default 0.95.
    self_discharge_per_hour : float
        Fractional self-discharge rate per hour [1/h].  Default 0.0.
    max_delta_p_kw_per_h : float or None
        Maximum allowed power ramp between consecutive steps [kW/h].
        Set to None to disable ramp-rate enforcement.
    min_rest_after_full_charge_h : float
        Mandatory idle period after reaching soc_max [h].  Default 0.0.
    min_rest_after_full_discharge_h : float
        Mandatory idle period after reaching soc_min [h].  Default 0.0.
    """

    energy_capacity_kwh: float
    p_charge_max_kw: float
    p_discharge_max_kw: float

    soc_min: float = 0.05
    soc_max: float = 0.95
    soc_initial: float = 0.50

    eta_charge: float = 0.95
    eta_discharge: float = 0.95

    self_discharge_per_hour: float = 0.0

    max_delta_p_kw_per_h: Optional[float] = None

    min_rest_after_full_charge_h: float = 0.0
    min_rest_after_full_discharge_h: float = 0.0

    @property
    def usable_energy_kwh(self) -> float:
        """
        Usable energy between soc_min and soc_max [kWh].

        Returns
        -------
        float
            (soc_max - soc_min) * energy_capacity_kwh
        """
        return (self.soc_max - self.soc_min) * self.energy_capacity_kwh

    def to_dict(self) -> Dict[str, Any]:
        """Return all parameters as a plain dictionary."""
        return asdict(self)


@dataclass
class BESSState:
    """
    Mutable run-time state of the BESS at a single point in time.

    Attributes
    ----------
    soc : float
        Current state of charge as a fraction [0, 1].
    prev_power_kw : float
        Battery power applied at the previous time step [kW].
        Required for ramp-rate enforcement.  Default 0.0.
    rest_remaining_h : float
        Remaining mandatory idle time [h].
        While this is > 0 the battery is forced to zero power.
    rest_reason : str
        Cause of the current idle period.
        One of {"none", "after_full_charge", "after_full_discharge"}.
    """

    soc: float
    prev_power_kw: float = 0.0
    rest_remaining_h: float = 0.0
    rest_reason: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        """Return the state as a plain dictionary."""
        return asdict(self)


# =========================================================
# 2. VALIDATION
# =========================================================

def validate_bess_params(params: BESSParams) -> None:
    """
    Verify that all BESS parameters are physically consistent.

    Parameters
    ----------
    params : BESSParams
        Parameter object to validate.

    Raises
    ------
    ValueError
        If any parameter is out of range or internally inconsistent.
    """
    if params.energy_capacity_kwh <= 0:
        raise ValueError("energy_capacity_kwh must be > 0")

    if params.p_charge_max_kw <= 0:
        raise ValueError("p_charge_max_kw must be > 0")

    if params.p_discharge_max_kw <= 0:
        raise ValueError("p_discharge_max_kw must be > 0")

    if not (0 <= params.soc_min < params.soc_max <= 1):
        raise ValueError("Required: 0 <= soc_min < soc_max <= 1")

    if not (params.soc_min <= params.soc_initial <= params.soc_max):
        raise ValueError("soc_initial must be within [soc_min, soc_max]")

    if not (0 < params.eta_charge <= 1):
        raise ValueError("eta_charge must be in (0, 1]")

    if not (0 < params.eta_discharge <= 1):
        raise ValueError("eta_discharge must be in (0, 1]")

    if not (0 <= params.self_discharge_per_hour < 1):
        raise ValueError("self_discharge_per_hour must be in [0, 1)")

    if params.max_delta_p_kw_per_h is not None and params.max_delta_p_kw_per_h <= 0:
        raise ValueError("max_delta_p_kw_per_h must be > 0 or None")

    if params.min_rest_after_full_charge_h < 0:
        raise ValueError("min_rest_after_full_charge_h must be >= 0")

    if params.min_rest_after_full_discharge_h < 0:
        raise ValueError("min_rest_after_full_discharge_h must be >= 0")


def validate_state(state: BESSState, params: BESSParams) -> None:
    """
    Verify that a BESSState is numerically valid and consistent with params.

    Parameters
    ----------
    state : BESSState
        State object to validate.
    params : BESSParams
        Associated battery parameters (used for context checks).

    Raises
    ------
    ValueError
        If any field is missing, NaN, or out of the expected range.
    """
    if not isinstance(state.soc, (int, float, np.floating)):
        raise ValueError("state.soc must be a number")

    if np.isnan(state.soc):
        raise ValueError("state.soc must not be NaN")

    if not isinstance(state.prev_power_kw, (int, float, np.floating)):
        raise ValueError("state.prev_power_kw must be a number")

    if np.isnan(state.prev_power_kw):
        raise ValueError("state.prev_power_kw must not be NaN")

    if not isinstance(state.rest_remaining_h, (int, float, np.floating)):
        raise ValueError("state.rest_remaining_h must be a number")

    if np.isnan(state.rest_remaining_h):
        raise ValueError("state.rest_remaining_h must not be NaN")

    if state.rest_remaining_h < 0:
        raise ValueError("state.rest_remaining_h must be >= 0")

    if not isinstance(state.rest_reason, str):
        raise ValueError("state.rest_reason must be a string")

    if state.rest_reason not in {"none", "after_full_charge", "after_full_discharge"}:
        raise ValueError(
            "state.rest_reason must be one of "
            "{'none', 'after_full_charge', 'after_full_discharge'}"
        )

    # SOC is not clipped here — only checked for gross validity.
    if not (0 <= state.soc <= 1):
        raise ValueError("state.soc must be in [0, 1]")


# =========================================================
# 3. SOC / ENERGY HELPER FUNCTIONS
# =========================================================

def clip_soc(soc: float, params: BESSParams) -> float:
    """
    Clamp a SOC value to the feasible interval [soc_min, soc_max].

    Parameters
    ----------
    soc : float
        Raw SOC fraction to clamp.
    params : BESSParams
        Battery parameters providing soc_min and soc_max.

    Returns
    -------
    float
        SOC clamped to [params.soc_min, params.soc_max].
    """
    return min(max(float(soc), params.soc_min), params.soc_max)


def soc_to_energy_kwh(soc: float, params: BESSParams) -> float:
    """
    Convert a SOC fraction to stored energy.

    Parameters
    ----------
    soc : float
        State of charge in [0, 1].
    params : BESSParams
        Battery parameters providing energy_capacity_kwh.

    Returns
    -------
    float
        Stored energy [kWh]  =  soc × energy_capacity_kwh.
    """
    return float(soc) * params.energy_capacity_kwh


def energy_to_soc(energy_kwh: float, params: BESSParams) -> float:
    """
    Convert stored energy to a SOC fraction.

    Parameters
    ----------
    energy_kwh : float
        Stored energy [kWh].
    params : BESSParams
        Battery parameters providing energy_capacity_kwh.

    Returns
    -------
    float
        State of charge  =  energy_kwh / energy_capacity_kwh.
    """
    return float(energy_kwh) / params.energy_capacity_kwh


def make_initial_state(
    params: BESSParams,
    soc: Optional[float] = None,
    prev_power_kw: float = 0.0,
    rest_remaining_h: float = 0.0,
    rest_reason: str = "none"
) -> BESSState:
    """
    Construct and validate an initial BESSState.

    Parameters
    ----------
    params : BESSParams
        Battery parameters; soc_initial is used when soc is None.
    soc : float, optional
        Override for the initial SOC.  If None, params.soc_initial is used.
    prev_power_kw : float, optional
        Power applied in the hypothetical step before t=0.  Default 0.0.
    rest_remaining_h : float, optional
        Remaining mandatory rest at t=0 [h].  Default 0.0.
    rest_reason : str, optional
        Reason for initial rest period.  Default "none".

    Returns
    -------
    BESSState
        Validated initial state with SOC clamped to [soc_min, soc_max].

    Raises
    ------
    ValueError
        If params or the resulting state fail validation.
    """
    validate_bess_params(params)

    if soc is None:
        soc = params.soc_initial

    state = BESSState(
        soc=clip_soc(soc, params),
        prev_power_kw=prev_power_kw,
        rest_remaining_h=float(rest_remaining_h),
        rest_reason=rest_reason
    )
    validate_state(state, params)
    return state


# =========================================================
# 4. POWER CONSTRAINT CALCULATION
# =========================================================

def get_power_limits_kw(
    state: BESSState,
    params: BESSParams,
    dt_h: float = 1.0
) -> Dict[str, float]:
    """
    Compute the feasible power interval [p_min_kw, p_max_kw] for the next step.

    Four constraints are intersected in order:

    1. **Inverter / PCS limits**
       -p_charge_max_kw  ≤  p  ≤  +p_discharge_max_kw

    2. **SOC / energy limits**

       Maximum charge power (so SOC does not exceed soc_max):

           |P_charge|_max = (E_max - E_now) / (η_charge · Δt)

       Maximum discharge power (so SOC does not fall below soc_min):

           P_discharge_max = (E_now - E_min) · η_discharge / Δt

    3. **Ramp-rate limit** (if max_delta_p_kw_per_h is set):

           |p(t) - p(t-1)| ≤ max_delta_p_kw_per_h · Δt

    4. **Mandatory rest lock** — if rest_remaining_h > 0, forces p = 0.

    Parameters
    ----------
    state : BESSState
        Current battery state.
    params : BESSParams
        Battery configuration.
    dt_h : float, optional
        Time-step duration [h].  Default 1.0.

    Returns
    -------
    dict
        Keys:

        - ``p_min_kw``                 — final lower bound
        - ``p_max_kw``                 — final upper bound
        - ``p_min_power_only_kw``      — inverter lower bound
        - ``p_max_power_only_kw``      — inverter upper bound
        - ``p_min_energy_only_kw``     — SOC-based lower bound
        - ``p_max_energy_only_kw``     — SOC-based upper bound
        - ``p_min_ramp_kw``            — ramp-rate lower bound
        - ``p_max_ramp_kw``            — ramp-rate upper bound
        - ``rest_lock_active``         — bool, True if rest is enforced
        - ``rest_remaining_h_start``   — rest timer at start of step

    Raises
    ------
    ValueError
        If params, state, or dt_h are invalid.
    """
    validate_bess_params(params)
    validate_state(state, params)

    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    soc = clip_soc(state.soc, params)
    e_now = soc_to_energy_kwh(soc, params)

    e_min = params.soc_min * params.energy_capacity_kwh
    e_max = params.soc_max * params.energy_capacity_kwh

    # ------------------------------------------------------------------
    # 1) Inverter / PCS limits
    # ------------------------------------------------------------------
    p_min_power_only_kw = -params.p_charge_max_kw
    p_max_power_only_kw = params.p_discharge_max_kw

    # ------------------------------------------------------------------
    # 2) SOC / energy limits
    #
    # Charge (negative sign convention):
    #   Charging at |P| kW for dt_h hours increases internal energy by
    #   |P| · η_charge · dt_h.  To stay at or below E_max:
    #       |P_charge|_max = (E_max - E_now) / (η_charge · dt_h)
    #
    # Discharge (positive sign convention):
    #   Discharging at P kW for dt_h hours removes P/η_discharge · dt_h
    #   from the battery.  To stay at or above E_min:
    #       P_discharge_max = (E_now - E_min) · η_discharge / dt_h
    # ------------------------------------------------------------------
    charge_limit_kw_by_energy = (e_max - e_now) / (params.eta_charge * dt_h)
    p_min_energy_only_kw = -max(0.0, charge_limit_kw_by_energy)

    discharge_limit_kw_by_energy = (e_now - e_min) * params.eta_discharge / dt_h
    p_max_energy_only_kw = max(0.0, discharge_limit_kw_by_energy)

    # Intersect inverter and energy constraints
    p_min_base_kw = max(p_min_power_only_kw, p_min_energy_only_kw)
    p_max_base_kw = min(p_max_power_only_kw, p_max_energy_only_kw)

    # ------------------------------------------------------------------
    # 3) Ramp-rate constraint (optional)
    #
    # Limits how fast the power can change between consecutive steps:
    #   p(t-1) - Δp_max · dt  ≤  p(t)  ≤  p(t-1) + Δp_max · dt
    # ------------------------------------------------------------------
    if params.max_delta_p_kw_per_h is not None:
        p_min_ramp_kw = state.prev_power_kw - params.max_delta_p_kw_per_h * dt_h
        p_max_ramp_kw = state.prev_power_kw + params.max_delta_p_kw_per_h * dt_h

        p_min_after_ramp_kw = max(p_min_base_kw, p_min_ramp_kw)
        p_max_after_ramp_kw = min(p_max_base_kw, p_max_ramp_kw)
    else:
        p_min_ramp_kw = -np.inf
        p_max_ramp_kw = np.inf
        p_min_after_ramp_kw = p_min_base_kw
        p_max_after_ramp_kw = p_max_base_kw

    # ------------------------------------------------------------------
    # 4) Mandatory rest lock
    #
    # If rest_remaining_h > 0 the only admissible power is zero.
    # ------------------------------------------------------------------
    rest_lock_active = state.rest_remaining_h > 1e-12

    if rest_lock_active:
        p_min_kw = 0.0
        p_max_kw = 0.0
    else:
        p_min_kw = p_min_after_ramp_kw
        p_max_kw = p_max_after_ramp_kw

    # Safety fallback: if constraints produce an empty interval, resolve
    # conservatively by collapsing to the nearest feasible point to zero.
    if p_min_kw > p_max_kw:
        if p_min_kw <= 0 <= p_max_kw:
            p_safe_kw = 0.0
        else:
            p_safe_kw = min(max(0.0, p_min_kw), p_max_kw)

        p_min_kw = p_safe_kw
        p_max_kw = p_safe_kw

    return {
        "p_min_kw": p_min_kw,
        "p_max_kw": p_max_kw,

        "p_min_power_only_kw": p_min_power_only_kw,
        "p_max_power_only_kw": p_max_power_only_kw,

        "p_min_energy_only_kw": p_min_energy_only_kw,
        "p_max_energy_only_kw": p_max_energy_only_kw,

        "p_min_ramp_kw": p_min_ramp_kw,
        "p_max_ramp_kw": p_max_ramp_kw,

        "rest_lock_active": rest_lock_active,
        "rest_remaining_h_start": float(state.rest_remaining_h),
    }


# =========================================================
# 5. COMMAND CLIPPING
# =========================================================

def clip_power_command_kw(
        p_cmd_kw: float,
        state: BESSState,
        params: BESSParams,
        dt_h: float = 1.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Project a requested power command onto the feasible interval.

    If the command is physically unreachable (SOC limit, power limit,
    ramp-rate, or mandatory rest), the command is silently clipped to
    the nearest feasible value.

    Example
    -------
    Controller requests +7 000 kW, but only +3 200 kW is available
    due to low SOC → p_applied_kw = +3 200 kW.

    Parameters
    ----------
    p_cmd_kw : float
        Requested battery power [kW].
    state : BESSState
        Current battery state.
    params : BESSParams
        Battery configuration.
    dt_h : float, optional
        Time-step duration [h].  Default 1.0.

    Returns
    -------
    p_applied_kw : float
        Feasible power actually applied [kW].
    info : dict
        Diagnostic dictionary with keys:

        - ``p_cmd_kw``         — original command
        - ``p_applied_kw``     — clipped command
        - ``was_clipped``      — bool
        - ``hit_min_limit``    — bool, True if clipped at lower bound
        - ``hit_max_limit``    — bool, True if clipped at upper bound
        - ``rest_lock_active`` — bool
        - ``hit_rest_lock``    — bool, True if command was non-zero during rest
        - (all keys from get_power_limits_kw)

    Raises
    ------
    ValueError
        If p_cmd_kw is not a finite number.
    """
    if not isinstance(p_cmd_kw, (int, float, np.floating)):
        raise ValueError("p_cmd_kw must be a number")

    if np.isnan(p_cmd_kw):
        raise ValueError("p_cmd_kw must not be NaN")

    limits = get_power_limits_kw(state=state, params=params, dt_h=dt_h)

    p_min_kw = limits["p_min_kw"]
    p_max_kw = limits["p_max_kw"]

    p_applied_kw = min(max(float(p_cmd_kw), p_min_kw), p_max_kw)

    rest_lock_active = bool(limits["rest_lock_active"])

    info = {
        "p_cmd_kw": float(p_cmd_kw),
        "p_applied_kw": p_applied_kw,

        "was_clipped": not np.isclose(float(p_cmd_kw), p_applied_kw),
        "hit_min_limit": float(p_cmd_kw) < p_min_kw,
        "hit_max_limit": float(p_cmd_kw) > p_max_kw,

        "rest_lock_active": rest_lock_active,
        "hit_rest_lock": rest_lock_active and not np.isclose(float(p_cmd_kw), 0.0),

        **limits,
    }

    return p_applied_kw, info


# =========================================================
# 6. SOC AND STATE UPDATE
# =========================================================

def update_soc(
        state: BESSState,
        p_bess_kw: float,
        params: BESSParams,
        dt_h: float = 1.0
) -> BESSState:
    """
    Advance the battery state by one time step given an applied power.

    Energy balance equations
    ------------------------
    Discharge  (p_bess_kw ≥ 0):

        E(t+1) = E(t) - (p_bess_kw / η_discharge) · dt_h

    Charge  (p_bess_kw < 0):

        E(t+1) = E(t) + |p_bess_kw| · η_charge · dt_h

    E(t+1) is then hard-clamped to [E_min, E_max].

    Mandatory rest timer
    --------------------
    - If the battery was already in a rest period, the timer is
      decremented by dt_h.
    - If the battery reaches soc_max during a charge step, a new
      mandatory rest of min_rest_after_full_charge_h hours is started.
    - If the battery reaches soc_min during a discharge step, a new
      mandatory rest of min_rest_after_full_discharge_h hours is started.

    Note on discrete-time rest
    --------------------------
    With dt_h = 1.0 and a rest duration of, say, 1.5 h, the model
    behaves conservatively: the next whole-hour step is also locked.

    Parameters
    ----------
    state : BESSState
        Battery state at the start of the step.
    p_bess_kw : float
        Power applied during the step [kW].
    params : BESSParams
        Battery configuration.
    dt_h : float, optional
        Time-step duration [h].  Default 1.0.

    Returns
    -------
    BESSState
        Updated battery state at the end of the step.

    Raises
    ------
    ValueError
        If inputs fail validation.
    """
    validate_bess_params(params)
    validate_state(state, params)

    if dt_h <= 0:
        raise ValueError("dt_h must be > 0")

    if not isinstance(p_bess_kw, (int, float, np.floating)):
        raise ValueError("p_bess_kw must be a number")

    if np.isnan(p_bess_kw):
        raise ValueError("p_bess_kw must not be NaN")

    soc_now = clip_soc(state.soc, params)
    e_now = soc_to_energy_kwh(soc_now, params)

    # Apply self-discharge before the commanded power
    if params.self_discharge_per_hour > 0:
        e_now = e_now * max(0.0, 1.0 - params.self_discharge_per_hour * dt_h)

    # Update internal energy
    if p_bess_kw >= 0:
        # Discharge: battery loses energy; output divided by efficiency
        e_next = e_now - (float(p_bess_kw) / params.eta_discharge) * dt_h
    else:
        # Charge: battery gains energy; input multiplied by efficiency
        e_next = e_now + (abs(float(p_bess_kw)) * params.eta_charge) * dt_h

    # Hard clamp to feasible energy range
    e_min = params.soc_min * params.energy_capacity_kwh
    e_max = params.soc_max * params.energy_capacity_kwh
    e_next = min(max(e_next, e_min), e_max)

    soc_next = energy_to_soc(e_next, params)
    soc_next = clip_soc(soc_next, params)

    # ------------------------------------------------------------------
    # Update mandatory rest timer
    # ------------------------------------------------------------------
    # 1) Decrement any existing rest timer.
    rest_remaining_h_after_decay = max(0.0, float(state.rest_remaining_h) - dt_h)

    if rest_remaining_h_after_decay > 1e-12:
        # Rest continues from the previous step
        next_rest_remaining_h = rest_remaining_h_after_decay
        next_rest_reason = state.rest_reason
    else:
        # Rest has expired (or was never active)
        next_rest_remaining_h = 0.0
        next_rest_reason = "none"

        # 2) Check whether this step triggers a new mandatory rest.
        reached_soc_max_this_step = (float(p_bess_kw) < 0) and np.isclose(soc_next, params.soc_max)
        reached_soc_min_this_step = (float(p_bess_kw) > 0) and np.isclose(soc_next, params.soc_min)

        if reached_soc_max_this_step and params.min_rest_after_full_charge_h > 0:
            next_rest_remaining_h = float(params.min_rest_after_full_charge_h)
            next_rest_reason = "after_full_charge"

        elif reached_soc_min_this_step and params.min_rest_after_full_discharge_h > 0:
            next_rest_remaining_h = float(params.min_rest_after_full_discharge_h)
            next_rest_reason = "after_full_discharge"

    return BESSState(
        soc=soc_next,
        prev_power_kw=float(p_bess_kw),
        rest_remaining_h=next_rest_remaining_h,
        rest_reason=next_rest_reason
    )


# =========================================================
# 7. SINGLE TIME-STEP SIMULATION
# =========================================================

def apply_bess_action(
        actual_kw: float,
        forecast_kw: float,
        state: BESSState,
        p_cmd_kw: float,
        params: BESSParams,
        dt_h: float = 1.0
) -> Tuple[Dict[str, Any], BESSState]:
    """
    Simulate one time step: clip a power command, update the battery,
    and compute all relevant metrics.

    Processing steps
    ----------------
    1. Compute deviation before BESS:

           deviation_before = actual_kw - forecast_kw

    2. Clip the command to the feasible power interval:

           p_applied_kw  ←  clip_power_command_kw(p_cmd_kw, ...)

    3. Compute net generation after BESS:

           actual_with_bess = actual_kw + p_applied_kw

       Sign rationale: positive p_applied_kw (discharge) adds power
       to the bus; negative p_applied_kw (charge) removes power.

    4. Compute deviation after BESS:

           deviation_after = actual_with_bess - forecast_kw

    5. Update SOC and the rest timer via update_soc().

    Parameters
    ----------
    actual_kw : float
        Actual generation/consumption before BESS [kW].
    forecast_kw : float
        Scheduled/forecast value [kW].
    state : BESSState
        Battery state at the start of the step.
    p_cmd_kw : float
        Requested battery power [kW].
    params : BESSParams
        Battery configuration.
    dt_h : float, optional
        Time-step duration [h].  Default 1.0.

    Returns
    -------
    result : dict
        Step-level metrics (see source for full key list), including
        soc_start, soc_end, deviation_before, p_bess_kw,
        actual_with_bess, deviation_after_bess, charge/discharge
        energy, clipping flags, and rest-timer diagnostics.
    next_state : BESSState
        Updated battery state for the following step.

    Raises
    ------
    ValueError
        If any numeric input is non-finite or of the wrong type.
    """
    if not isinstance(actual_kw, (int, float, np.floating)):
        raise ValueError("actual_kw must be a number")
    if not isinstance(forecast_kw, (int, float, np.floating)):
        raise ValueError("forecast_kw must be a number")
    if np.isnan(actual_kw) or np.isnan(forecast_kw):
        raise ValueError("actual_kw and forecast_kw must not be NaN")

    deviation_before_kw = float(actual_kw) - float(forecast_kw)

    rest_remaining_h_start = float(state.rest_remaining_h)
    rest_reason_start = state.rest_reason
    rest_lock_active_start = rest_remaining_h_start > 1e-12

    # Step 1: Clip command to physical constraints
    p_applied_kw, clip_info = clip_power_command_kw(
        p_cmd_kw=p_cmd_kw,
        state=state,
        params=params,
        dt_h=dt_h
    )

    # Step 2: Net generation after BESS
    actual_with_bess_kw = float(actual_kw) + p_applied_kw
    deviation_after_kw = actual_with_bess_kw - float(forecast_kw)

    # Step 3: Update battery state
    next_state = update_soc(
        state=state,
        p_bess_kw=p_applied_kw,
        params=params,
        dt_h=dt_h
    )

    # Diagnostic flags for SOC boundary events
    reached_soc_max_this_step = (p_applied_kw < 0) and np.isclose(next_state.soc, params.soc_max)
    reached_soc_min_this_step = (p_applied_kw > 0) and np.isclose(next_state.soc, params.soc_min)

    # True if mandatory rest was newly triggered on this step
    rest_started_this_step = (
        next_state.rest_remaining_h > max(0.0, rest_remaining_h_start - dt_h) + 1e-12
    )

    # Separate charge / discharge power for energy accounting
    if p_applied_kw >= 0:
        charge_power_kw = 0.0
        discharge_power_kw = p_applied_kw
    else:
        charge_power_kw = abs(p_applied_kw)
        discharge_power_kw = 0.0

    charge_energy_input_kwh = charge_power_kw * dt_h
    discharge_energy_output_kwh = discharge_power_kw * dt_h

    result = {
        "soc_start": state.soc,
        "soc_end": next_state.soc,

        "prev_power_kw": state.prev_power_kw,

        "deviation_before": deviation_before_kw,
        "p_cmd_kw": float(p_cmd_kw),
        "p_bess_kw": p_applied_kw,
        "actual_with_bess": actual_with_bess_kw,
        "deviation_after_bess": deviation_after_kw,

        "charge_power_kw": charge_power_kw,
        "discharge_power_kw": discharge_power_kw,
        "charge_energy_input_kwh": charge_energy_input_kwh,
        "discharge_energy_output_kwh": discharge_energy_output_kwh,

        "bess_was_clipped": clip_info["was_clipped"],
        "bess_hit_min_limit": clip_info["hit_min_limit"],
        "bess_hit_max_limit": clip_info["hit_max_limit"],

        "bess_rest_lock_active_start": rest_lock_active_start,
        "bess_hit_rest_lock": clip_info["hit_rest_lock"],
        "bess_rest_remaining_h_start": rest_remaining_h_start,
        "bess_rest_remaining_h_end": next_state.rest_remaining_h,
        "bess_rest_reason_start": rest_reason_start,
        "bess_rest_reason_end": next_state.rest_reason,
        "bess_rest_started_this_step": rest_started_this_step,

        "bess_reached_soc_max_this_step": reached_soc_max_this_step,
        "bess_reached_soc_min_this_step": reached_soc_min_this_step,

        "bess_p_min_kw": clip_info["p_min_kw"],
        "bess_p_max_kw": clip_info["p_max_kw"],

        "bess_p_min_power_only_kw": clip_info["p_min_power_only_kw"],
        "bess_p_max_power_only_kw": clip_info["p_max_power_only_kw"],
        "bess_p_min_energy_only_kw": clip_info["p_min_energy_only_kw"],
        "bess_p_max_energy_only_kw": clip_info["p_max_energy_only_kw"],
        "bess_p_min_ramp_kw": clip_info["p_min_ramp_kw"],
        "bess_p_max_ramp_kw": clip_info["p_max_ramp_kw"],
    }

    return result, next_state


# =========================================================
# 8. BUILT-IN GREEDY CONTROLLER
# =========================================================

def greedy_deviation_controller(
        row: pd.Series,
        state: BESSState,
        params: BESSParams
) -> float:
    """
    Rule-based baseline: attempt to fully compensate the current deviation.

    Derivation
    ----------
    The deviation after BESS is:

        deviation_after = (actual + p_bess) - forecast
                        = deviation_before + p_bess

    Setting deviation_after = 0 gives:

        p_bess = -deviation_before = -(actual - forecast)

    This is a greedy single-step strategy and makes no assumptions about
    future time steps.  It serves as a lower-bound benchmark for more
    sophisticated controllers.

    Physical safety
    ---------------
    Even if this function returns a non-zero command, the physical layer
    (clip_power_command_kw) will override it with p = 0 when a
    mandatory rest period is active.

    Parameters
    ----------
    row : pd.Series
        Current data row.  Must contain "actual" and "forecast" fields.
    state : BESSState
        Current battery state (unused by this controller, but required
        by the controller interface).
    params : BESSParams
        Battery configuration (unused here, present for interface
        compatibility).

    Returns
    -------
    float
        Requested power command [kW].

    Raises
    ------
    ValueError
        If "actual" or "forecast" are not present in row.
    """
    if "actual" not in row.index or "forecast" not in row.index:
        raise ValueError(
            "greedy_deviation_controller requires 'actual' and 'forecast' columns"
        )

    deviation_kw = float(row["actual"]) - float(row["forecast"])
    return -deviation_kw


# =========================================================
# 9. MULTI-STEP SIMULATION — ACTION ARRAY
# =========================================================

def simulate_with_actions(
        df: pd.DataFrame,
        actions_kw: List[float],
        params: BESSParams,
        dt_h: float = 1.0,
        initial_state: Optional[BESSState] = None,
        actual_col: str = "actual",
        forecast_col: str = "forecast"
) -> pd.DataFrame:
    """
    Simulate BESS over a full horizon given a pre-computed power schedule.

    This is the primary entry point for offline optimization:

        optimizer.py  →  find optimal [p_1, …, p_T]
        bess_model.py →  simulate_with_actions([p_1, …, p_T])

    The physical model still clips each command to the feasible interval,
    so the output reflects what is actually achievable.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series.  Must contain actual_col and forecast_col.
    actions_kw : list of float
        Requested BESS power for each row [kW].  Must have the same
        length as df.
    params : BESSParams
        Battery configuration.
    dt_h : float, optional
        Time-step duration [h].  Default 1.0.
    initial_state : BESSState, optional
        Starting battery state.  If None, created from params.soc_initial.
    actual_col : str, optional
        Name of the actual-generation column.  Default "actual".
    forecast_col : str, optional
        Name of the forecast column.  Default "forecast".

    Returns
    -------
    pd.DataFrame
        Copy of df with all simulation result columns appended
        (same column set as apply_bess_action result dict).

    Raises
    ------
    ValueError
        If required columns are missing or actions_kw length mismatches df.
    """
    validate_bess_params(params)

    if actual_col not in df.columns:
        raise ValueError(f"Column '{actual_col}' not found in df")
    if forecast_col not in df.columns:
        raise ValueError(f"Column '{forecast_col}' not found in df")

    if len(actions_kw) != len(df):
        raise ValueError("Length of actions_kw must match the number of rows in df")

    if initial_state is None:
        state = make_initial_state(params)
    else:
        validate_state(initial_state, params)
        state = BESSState(
            soc=clip_soc(initial_state.soc, params),
            prev_power_kw=initial_state.prev_power_kw,
            rest_remaining_h=initial_state.rest_remaining_h,
            rest_reason=initial_state.rest_reason
        )

    df_out = df.copy()
    results = []

    for i, (_, row) in enumerate(df_out.iterrows()):
        actual_kw = row[actual_col]
        forecast_kw = row[forecast_col]
        p_cmd_kw = float(actions_kw[i])

        result, state = apply_bess_action(
            actual_kw=actual_kw,
            forecast_kw=forecast_kw,
            state=state,
            p_cmd_kw=p_cmd_kw,
            params=params,
            dt_h=dt_h
        )

        results.append(result)

    results_df = pd.DataFrame(results, index=df_out.index)
    df_out = pd.concat([df_out, results_df], axis=1)

    return df_out


# =========================================================
# 10. MULTI-STEP SIMULATION — CONTROLLER FUNCTION
# =========================================================

def simulate_with_controller(
        df: pd.DataFrame,
        controller: Callable[[pd.Series, BESSState, BESSParams], float],
        params: BESSParams,
        dt_h: float = 1.0,
        initial_state: Optional[BESSState] = None,
        actual_col: str = "actual",
        forecast_col: str = "forecast"
) -> pd.DataFrame:
    """
    Simulate BESS over a full horizon using a callable controller.

    At each time step the controller is called with:

        p_cmd_kw = controller(row, state, params)

    and the physical model clips the command, updates the state,
    and records the results.

    Controller interface
    --------------------
    Any callable with the signature::

        def my_controller(
            row: pd.Series,
            state: BESSState,
            params: BESSParams
        ) -> float: ...

    Examples of compatible controllers:
    - greedy_deviation_controller (built-in, see section 8)
    - Reinforcement-learning policy
    - Model-predictive controller (receding-horizon)
    - Any rule-based heuristic

    Parameters
    ----------
    df : pd.DataFrame
        Input time series.  Must contain actual_col and forecast_col.
    controller : callable
        Strategy function returning a power command [kW].
    params : BESSParams
        Battery configuration.
    dt_h : float, optional
        Time-step duration [h].  Default 1.0.
    initial_state : BESSState, optional
        Starting battery state.  If None, created from params.soc_initial.
    actual_col : str, optional
        Name of the actual-generation column.  Default "actual".
    forecast_col : str, optional
        Name of the forecast column.  Default "forecast".

    Returns
    -------
    pd.DataFrame
        Copy of df with all simulation result columns appended.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    validate_bess_params(params)

    if actual_col not in df.columns:
        raise ValueError(f"Column '{actual_col}' not found in df")
    if forecast_col not in df.columns:
        raise ValueError(f"Column '{forecast_col}' not found in df")

    if initial_state is None:
        state = make_initial_state(params)
    else:
        validate_state(initial_state, params)
        state = BESSState(
            soc=clip_soc(initial_state.soc, params),
            prev_power_kw=initial_state.prev_power_kw,
            rest_remaining_h=initial_state.rest_remaining_h,
            rest_reason=initial_state.rest_reason
        )

    df_out = df.copy()
    results = []

    for _, row_original in df_out.iterrows():
        row = row_original.copy()

        # Provide standard aliases if non-default column names are used
        if actual_col != "actual":
            row["actual"] = row_original[actual_col]
        if forecast_col != "forecast":
            row["forecast"] = row_original[forecast_col]

        p_cmd_kw = controller(row, state, params)

        actual_value = row_original[actual_col]
        forecast_value = row_original[forecast_col]

        result, state = apply_bess_action(
            actual_kw=actual_value,
            forecast_kw=forecast_value,
            state=state,
            p_cmd_kw=p_cmd_kw,
            params=params,
            dt_h=dt_h
        )

        results.append(result)

    results_df = pd.DataFrame(results, index=df_out.index)
    df_out = pd.concat([df_out, results_df], axis=1)

    return df_out


# =========================================================
# 11. SUMMARY METRICS
# =========================================================

def summarize_bess_results(df_bess: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute aggregate performance metrics from a simulation output DataFrame.

    Expected columns (all optional — missing columns yield None):

    - ``p_bess_kw``
    - ``charge_energy_input_kwh``
    - ``discharge_energy_output_kwh``
    - ``deviation_before``
    - ``deviation_after_bess``
    - ``soc_end``
    - ``bess_was_clipped``
    - ``bess_rest_lock_active_start``
    - ``bess_rest_started_this_step``
    - ``bess_reached_soc_max_this_step``
    - ``bess_reached_soc_min_this_step``

    Equivalent-cycle calculation
    ----------------------------
    A simplified engineering estimate:

        throughput_kwh = total_charge_in + total_discharge_out
        equivalent_cycles = throughput_kwh / (2 · usable_energy_kwh)

    Note: usable_energy_kwh is not computed here to keep this function
    independent of BESSParams.  throughput_kwh is returned so the caller
    can compute equivalent cycles externally.

    Parameters
    ----------
    df_bess : pd.DataFrame
        Output of simulate_with_actions or simulate_with_controller.

    Returns
    -------
    dict
        Keys:

        - ``rows``                             — number of time steps
        - ``total_charge_energy_input_kwh``    — sum of charge energy
        - ``total_discharge_energy_output_kwh``— sum of discharge energy
        - ``throughput_kwh``                   — total energy throughput
        - ``mean_abs_deviation_before``        — MAD before BESS [kW]
        - ``mean_abs_deviation_after``         — MAD after BESS [kW]
        - ``final_soc``                        — SOC at the last step
        - ``clipped_hours``                    — steps where command was clipped
        - ``rest_locked_hours``                — steps where rest lock was active
        - ``rest_starts_count``                — number of new rest periods triggered
        - ``full_charge_events``               — times soc_max was reached
        - ``full_discharge_events``            — times soc_min was reached
    """
    summary = {}

    def safe_sum(col: str):
        return float(df_bess[col].sum()) if col in df_bess.columns else None

    def safe_last(col: str):
        return float(df_bess[col].iloc[-1]) if col in df_bess.columns and len(df_bess) > 0 else None

    summary["rows"] = int(len(df_bess))
    summary["total_charge_energy_input_kwh"] = safe_sum("charge_energy_input_kwh")
    summary["total_discharge_energy_output_kwh"] = safe_sum("discharge_energy_output_kwh")

    if (
        "charge_energy_input_kwh" in df_bess.columns
        and "discharge_energy_output_kwh" in df_bess.columns
        and len(df_bess) > 0
    ):
        throughput_kwh = float(
            df_bess["charge_energy_input_kwh"].sum() +
            df_bess["discharge_energy_output_kwh"].sum()
        )
        summary["throughput_kwh"] = throughput_kwh
    else:
        summary["throughput_kwh"] = None

    summary["mean_abs_deviation_before"] = (
        float(df_bess["deviation_before"].abs().mean())
        if "deviation_before" in df_bess.columns else None
    )

    summary["mean_abs_deviation_after"] = (
        float(df_bess["deviation_after_bess"].abs().mean())
        if "deviation_after_bess" in df_bess.columns else None
    )

    summary["final_soc"] = safe_last("soc_end")

    if "bess_was_clipped" in df_bess.columns:
        summary["clipped_hours"] = int(df_bess["bess_was_clipped"].sum())
    else:
        summary["clipped_hours"] = None

    if "bess_rest_lock_active_start" in df_bess.columns:
        summary["rest_locked_hours"] = int(df_bess["bess_rest_lock_active_start"].sum())
    else:
        summary["rest_locked_hours"] = None

    if "bess_rest_started_this_step" in df_bess.columns:
        summary["rest_starts_count"] = int(df_bess["bess_rest_started_this_step"].sum())
    else:
        summary["rest_starts_count"] = None

    if "bess_reached_soc_max_this_step" in df_bess.columns:
        summary["full_charge_events"] = int(df_bess["bess_reached_soc_max_this_step"].sum())
    else:
        summary["full_charge_events"] = None

    if "bess_reached_soc_min_this_step" in df_bess.columns:
        summary["full_discharge_events"] = int(df_bess["bess_reached_soc_min_this_step"].sum())
    else:
        summary["full_discharge_events"] = None

    return summary


# =========================================================
# 12. FAST DP KERNEL (NO VALIDATION)
# =========================================================

def apply_bess_action_fast(
    soc: float,
    rest_remaining_h: float,
    rest_reason: str,
    p_cmd_kw: float,
    p_charge_max_kw: float,
    p_discharge_max_kw: float,
    soc_min: float,
    soc_max: float,
    eta_charge: float,
    eta_discharge: float,
    energy_capacity_kwh: float,
    min_rest_after_full_charge_h: float,
    min_rest_after_full_discharge_h: float,
    dt_h: float = 1.0,
) -> Tuple[float, float, float, str, float]:
    """
    Minimal BESS step kernel for use inside dynamic-programming solvers.

    This function replicates the logic of apply_bess_action but omits
    all input validation, object creation, and dictionary allocation.
    Battery parameters are passed as plain scalars to avoid attribute
    lookups in tight inner loops.

    It is not intended for direct use in application code — use
    apply_bess_action for that purpose.

    Algorithm
    ---------
    1. If rest_remaining_h > 0 → force p_applied = 0.
    2. Otherwise clip p_cmd_kw to [p_min, p_max] derived from
       inverter limits and SOC energy limits.
    3. Update internal energy using charge/discharge efficiency.
    4. Clamp SOC to [soc_min, soc_max].
    5. Decrement rest timer; start a new rest if a SOC boundary
       was reached this step.
    6. Return applied power, next SOC, next rest timer, next rest
       reason, and energy throughput.

    Parameters
    ----------
    soc : float
        Current SOC fraction.
    rest_remaining_h : float
        Remaining mandatory rest [h].
    rest_reason : str
        Current rest reason string.
    p_cmd_kw : float
        Requested power [kW].
    p_charge_max_kw : float
        Maximum charge power (magnitude) [kW].
    p_discharge_max_kw : float
        Maximum discharge power [kW].
    soc_min : float
        Lower SOC bound.
    soc_max : float
        Upper SOC bound.
    eta_charge : float
        Charge efficiency.
    eta_discharge : float
        Discharge efficiency.
    energy_capacity_kwh : float
        Total nameplate energy [kWh].
    min_rest_after_full_charge_h : float
        Mandatory rest after full charge [h].
    min_rest_after_full_discharge_h : float
        Mandatory rest after full discharge [h].
    dt_h : float, optional
        Time-step duration [h].  Default 1.0.

    Returns
    -------
    p_applied : float
        Power actually applied after clipping [kW].
    soc_next : float
        SOC at end of step.
    next_rest_h : float
        Remaining rest at end of step [h].
    next_rest_reason : str
        Rest reason at end of step.
    throughput : float
        |p_applied| · dt_h  [kWh], used for degradation weighting.
    """
    # ------------------------------------------------------------------
    # Clip power command
    # ------------------------------------------------------------------
    if rest_remaining_h > 1e-12:
        p_applied = 0.0
    else:
        e_now = soc * energy_capacity_kwh
        e_min = soc_min * energy_capacity_kwh
        e_max = soc_max * energy_capacity_kwh

        charge_limit  = (e_max - e_now) / (eta_charge * dt_h)
        discharge_limit = (e_now - e_min) * eta_discharge / dt_h

        p_min = max(-p_charge_max_kw, -charge_limit)
        p_max = min( p_discharge_max_kw, discharge_limit)

        if p_min > p_max:
            p_applied = 0.0
        else:
            p_applied = min(max(p_cmd_kw, p_min), p_max)

    # ------------------------------------------------------------------
    # Update SOC
    # ------------------------------------------------------------------
    e_now = soc * energy_capacity_kwh
    if p_applied >= 0.0:
        e_next = e_now - (p_applied / eta_discharge) * dt_h
    else:
        e_next = e_now + (-p_applied * eta_charge) * dt_h

    e_min = soc_min * energy_capacity_kwh
    e_max = soc_max * energy_capacity_kwh
    e_next = min(max(e_next, e_min), e_max)
    soc_next = min(max(e_next / energy_capacity_kwh, soc_min), soc_max)

    # ------------------------------------------------------------------
    # Update rest timer
    # ------------------------------------------------------------------
    rest_after_decay = max(0.0, rest_remaining_h - dt_h)

    if rest_after_decay > 1e-12:
        next_rest_h = rest_after_decay
        next_rest_reason = rest_reason
    else:
        next_rest_h = 0.0
        next_rest_reason = "none"

        if p_applied < 0.0 and abs(soc_next - soc_max) < 1e-9:
            if min_rest_after_full_charge_h > 0:
                next_rest_h = min_rest_after_full_charge_h
                next_rest_reason = "after_full_charge"
        elif p_applied > 0.0 and abs(soc_next - soc_min) < 1e-9:
            if min_rest_after_full_discharge_h > 0:
                next_rest_h = min_rest_after_full_discharge_h
                next_rest_reason = "after_full_discharge"

    throughput = abs(p_applied) * dt_h

    return p_applied, soc_next, next_rest_h, next_rest_reason, throughput


# =========================================================
# QUICK MODULE TEST
# =========================================================

if __name__ == "__main__":
    from io_data import load_input_data

    file_path = "import/korem.xlsx"
    df, meta = load_input_data(file_path)

    params = BESSParams(
        energy_capacity_kwh=10_000,       # 10 MWh
        p_charge_max_kw=5_000,            # 5 MW charge
        p_discharge_max_kw=5_000,         # 5 MW discharge
        soc_min=0.10,
        soc_max=0.90,
        soc_initial=0.50,
        eta_charge=0.95,
        eta_discharge=0.95,
        self_discharge_per_hour=0.0,
        max_delta_p_kw_per_h=None,
        min_rest_after_full_charge_h=1.5,     # 90 min
        min_rest_after_full_discharge_h=1.5   # 90 min
    )

    df_bess = simulate_with_controller(
        df=df,
        controller=greedy_deviation_controller,
        params=params,
        dt_h=1.0,
        initial_state=None,
        actual_col="actual",
        forecast_col="forecast"
    )

    print("=== HEAD ===")
    cols_to_show = [
        "datetime", "forecast", "actual",
        "deviation_before", "p_cmd_kw", "p_bess_kw",
        "actual_with_bess", "deviation_after_bess",
        "soc_start", "soc_end",
        "bess_rest_lock_active_start",
        "bess_rest_remaining_h_start", "bess_rest_remaining_h_end",
        "bess_rest_started_this_step",
        "bess_reached_soc_max_this_step", "bess_reached_soc_min_this_step",
        "bess_was_clipped",
    ]
    existing = [c for c in cols_to_show if c in df_bess.columns]
    print(df_bess[existing].head(20))

    print("\n=== SUMMARY ===")
    for k, v in summarize_bess_results(df_bess).items():
        print(f"{k}: {v}")

    # Uncomment to export:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_bess.to_excel(f"export/output_after_bess_{ts}.xlsx", index=False, engine="openpyxl")
    