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
# 1. USER SETTINGS
# =========================================================

INPUT_FILE = "import/korem.xlsx"
SHEET_NAME = "Лист1"

# -----------------------------
# BESS scenario to test
# -----------------------------
BESS_PARAMS = BESSParams(
    energy_capacity_kwh=120000,
    p_charge_max_kw=30000,
    p_discharge_max_kw=30000,
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

# -----------------------------
# Offline DP settings
# ВАЖНО:
# - чем мельче шаги, тем точнее, но тем медленнее
# - для первого теста оставь так
# -----------------------------
RUN_OFFLINE_DP = True
DP_SOC_STEP = 0.05          # шаг SOC-сетки
DP_ACTION_STEP_KW = 1000.0  # шаг сетки действий
DP_REST_STEP_H = 0.5        # шаг дискретизации rest state
DP_TERMINAL_SOC_WEIGHT = 0.0
DP_DEGRADATION_WEIGHT_PER_KWH = 0.0

# Если хочешь ускорить тест, можно ограничить первые N часов:
# None = весь файл
LIMIT_HOURS: Optional[int] = None

# -----------------------------
# Export
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"export/run_optimizer_results_{timestamp}.xlsx"


# =========================================================
# 2. HELPERS
# =========================================================

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, float) and np.isnan(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _slug(prefix: str) -> str:
    s = str(prefix).strip().lower().replace(" ", "_").replace("-", "_")
    if not s.endswith("_"):
        s += "_"
    return s


def _require_meta(meta: Dict[str, Any]) -> None:
    required = [
        "tariff",
        "acceptable_range_plus",
        "acceptable_range_minus",
        "decreasing_factor",
        "increasing_factor",
    ]
    missing = [k for k in required if k not in meta or meta[k] is None]
    if missing:
        raise KeyError(f"В meta отсутствуют обязательные ключи: {missing}")


def _stage_loss_exact(forecast: float, fact: float, meta: Dict[str, Any]) -> float:
    """
    Точная часовая loss-функция, повторяющая economics.py,
    но без DataFrame overhead (важно для DP).
    """
    tariff = _safe_float(meta["tariff"])
    acceptable_range_plus = _safe_float(meta["acceptable_range_plus"])
    acceptable_range_minus = _safe_float(meta["acceptable_range_minus"])
    decreasing_factor = _safe_float(meta["decreasing_factor"])
    increasing_factor = _safe_float(meta["increasing_factor"])

    deviation = fact - forecast

    if forecast > 0:
        deviation_pct = deviation / forecast
    else:
        deviation_pct = 1.0 if fact != 0 else 0.0

    sales_forecast = forecast * tariff

    # positive deviation block
    positive_dev_pct = max(deviation_pct, 0.0)
    within_5_pct_positive = min(positive_dev_pct, acceptable_range_plus)

    if forecast > 0:
        sales_within_5pct = within_5_pct_positive * forecast * tariff
    else:
        sales_within_5pct = acceptable_range_plus * fact * tariff

    beyond_5_pct_positive = max(positive_dev_pct - acceptable_range_plus, 0.0)

    if forecast > 0:
        sales_beyond_5pct = beyond_5_pct_positive * forecast * tariff * decreasing_factor
    else:
        sales_beyond_5pct = beyond_5_pct_positive * fact * tariff * decreasing_factor

    # negative deviation block
    negative_dev_pct = min(deviation_pct, 0.0)
    within_5_pct_negative = max(negative_dev_pct, acceptable_range_minus)
    purchase_within_5pct = within_5_pct_negative * forecast * tariff

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
    loss = unpenalized_sales - total_sales_penalized
    return float(loss)


def _corridor_command(row: pd.Series, meta: Dict[str, Any]) -> float:
    """
    Возвращает p_cmd_kw, чтобы довести actual до ближайшей границы допустимого коридора.
    """
    actual = _safe_float(row["actual"])
    forecast = _safe_float(row["forecast"])
    acc_plus = _safe_float(meta["acceptable_range_plus"])
    acc_minus = _safe_float(meta["acceptable_range_minus"])

    if forecast > 0:
        lower_ok = forecast * (1.0 + acc_minus)
        upper_ok = forecast * (1.0 + acc_plus)
    else:
        lower_ok = forecast
        upper_ok = forecast

    if actual > upper_ok:
        return upper_ok - actual  # отрицательно -> заряд
    if actual < lower_ok:
        return lower_ok - actual  # положительно -> разряд
    return 0.0


def corridor_controller_factory(meta: Dict[str, Any]) -> Callable[[pd.Series, BESSState, BESSParams], float]:
    def controller(row: pd.Series, state: BESSState, params: BESSParams) -> float:
        return _corridor_command(row, meta)
    return controller


def evaluate_base_case(df: pd.DataFrame, meta: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
        "scenario": "no_bess",
        "optimizer_type": "base",
        "energy_capacity_kwh": 0.0,
        "p_charge_max_kw": 0.0,
        "p_discharge_max_kw": 0.0,
        "total_loss": float(econ[f"{prefix}loss"]),
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
        "scenario": scenario_name,
        "optimizer_type": "controller",
        **bess,
        "energy_capacity_kwh": params.energy_capacity_kwh,
        "p_charge_max_kw": params.p_charge_max_kw,
        "p_discharge_max_kw": params.p_discharge_max_kw,
        "total_loss": float(econ[f"{prefix}loss"]),
        **econ,
    }
    return df_out, summary


# =========================================================
# 3. OFFLINE DP
# =========================================================

def _build_action_grid(params: BESSParams, step_kw: float) -> List[float]:
    if step_kw <= 0:
        raise ValueError("DP_ACTION_STEP_KW должно быть > 0")

    neg = np.arange(-params.p_charge_max_kw, 0.0, step_kw)
    pos = np.arange(0.0, params.p_discharge_max_kw + step_kw, step_kw)

    actions = list(neg) + list(pos)
    actions = sorted(set(float(round(x, 10)) for x in actions))
    if 0.0 not in actions:
        actions.append(0.0)
        actions = sorted(actions)
    return actions


def _build_soc_grid(params: BESSParams, soc_step: float) -> List[float]:
    if soc_step <= 0:
        raise ValueError("DP_SOC_STEP должно быть > 0")

    vals = np.arange(params.soc_min, params.soc_max + soc_step / 2.0, soc_step)
    vals = np.clip(vals, params.soc_min, params.soc_max)
    vals = sorted(set(float(round(x, 10)) for x in vals))
    if vals[0] != params.soc_min:
        vals.insert(0, float(params.soc_min))
    if vals[-1] != params.soc_max:
        vals.append(float(params.soc_max))
    return vals


def _build_rest_grid(params: BESSParams, rest_step_h: float) -> List[float]:
    max_rest = max(
        _safe_float(getattr(params, "min_rest_after_full_charge_h", 0.0)),
        _safe_float(getattr(params, "min_rest_after_full_discharge_h", 0.0)),
    )
    vals = np.arange(0.0, max_rest + rest_step_h / 2.0, rest_step_h)
    vals = sorted(set(float(round(x, 10)) for x in vals))
    if 0.0 not in vals:
        vals.insert(0, 0.0)
    return vals


def _nearest_index(grid: List[float], value: float) -> int:
    arr = np.asarray(grid, dtype=float)
    return int(np.argmin(np.abs(arr - float(value))))


def _quantize_state(
    state: BESSState,
    soc_grid: List[float],
    power_grid: List[float],
    rest_grid: List[float],
) -> Tuple[int, int, int, int]:
    reason_map = {
        "none": 0,
        "after_full_charge": 1,
        "after_full_discharge": 2,
    }

    soc_i = _nearest_index(soc_grid, float(state.soc))
    p_i = _nearest_index(power_grid, float(state.prev_power_kw))
    r_i = _nearest_index(rest_grid, max(0.0, float(state.rest_remaining_h)))

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
    Полноценный offline DP:
    - знает весь исторический ряд заранее
    - ищет минимум суммарного loss
    - учитывает физику BESS через apply_bess_action()

    Выход:
    - optimal_actions_kw: список оптимальных p_cmd_kw по часам
    - optimal_objective: минимальное значение objective
    """
    if len(df) == 0:
        return [], 0.0

    actual_arr = df["actual"].astype(float).to_numpy()
    forecast_arr = df["forecast"].astype(float).to_numpy()
    T = len(df)

    action_grid = _build_action_grid(params, step_kw=action_step_kw)
    soc_grid = _build_soc_grid(params, soc_step=soc_step)
    power_grid = list(action_grid)
    rest_grid = _build_rest_grid(params, rest_step_h=rest_step_h)

    initial_state = make_initial_state(params)
    initial_key = _quantize_state(initial_state, soc_grid, power_grid, rest_grid)

    sys.setrecursionlimit(max(20000, T + 2000))

    @lru_cache(maxsize=None)
    def V(t: int, soc_i: int, p_i: int, r_i: int, reason_i: int) -> float:
        if t >= T:
            terminal_soc = soc_grid[soc_i]
            return float(terminal_soc_weight) * abs(terminal_soc - params.soc_initial)

        state = _decode_state((soc_i, p_i, r_i, reason_i), soc_grid, power_grid, rest_grid)

        best = np.inf
        actual_t = float(actual_arr[t])
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
                _safe_float(result.get("charge_energy_input_kwh", 0.0))
                + _safe_float(result.get("discharge_energy_output_kwh", 0.0))
            )
            stage_obj = stage_loss + float(degradation_weight_per_kwh) * throughput

            next_key = _quantize_state(next_state, soc_grid, power_grid, rest_grid)
            total = stage_obj + V(t + 1, *next_key)

            if total < best:
                best = total

        return float(best)

    # Forward reconstruction
    actions: List[float] = []
    cur_state = make_initial_state(params)

    for t in range(T):
        best_action = 0.0
        best_total = np.inf

        actual_t = float(actual_arr[t])
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
                _safe_float(result.get("charge_energy_input_kwh", 0.0))
                + _safe_float(result.get("discharge_energy_output_kwh", 0.0))
            )
            stage_obj = stage_loss + float(degradation_weight_per_kwh) * throughput

            next_key = _quantize_state(next_state, soc_grid, power_grid, rest_grid)
            total = stage_obj + V(t + 1, *next_key)

            if total < best_total:
                best_total = total
                best_action = float(p_cmd_kw)

        # apply chosen action to get next state
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

    # прогоняем actions через ту же физику (через controller-обертку)
    action_iter = iter(actions_kw)

    def controller_from_actions(row: pd.Series, state: BESSState, params_: BESSParams) -> float:
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
        "scenario": "offline_dp",
        "optimizer_type": "offline_dp",
        **bess,
        "energy_capacity_kwh": params.energy_capacity_kwh,
        "p_charge_max_kw": params.p_charge_max_kw,
        "p_discharge_max_kw": params.p_discharge_max_kw,
        "dp_objective": float(optimal_objective),
        "dp_soc_step": DP_SOC_STEP,
        "dp_action_step_kw": DP_ACTION_STEP_KW,
        "dp_rest_step_h": DP_REST_STEP_H,
        "total_loss": float(econ[f"{prefix}loss"]),
        **econ,
    }

    return df_out, summary


# =========================================================
# 4. MAIN
# =========================================================

def main() -> None:
    _require_meta  # just to keep linter happy

    df, meta = load_input_data(INPUT_FILE, sheet_name=SHEET_NAME)
    _require_meta(meta)

    if LIMIT_HOURS is not None:
        df = df.head(int(LIMIT_HOURS)).copy()

    # -------------------------
    # BASE
    # -------------------------
    df_base, summary_base = evaluate_base_case(df, meta)

    # -------------------------
    # GREEDY
    # -------------------------
    df_greedy, summary_greedy = evaluate_controller_case(
        df=df,
        meta=meta,
        params=BESS_PARAMS,
        controller=greedy_deviation_controller,
        scenario_name="greedy",
    )

    # -------------------------
    # CORRIDOR
    # -------------------------
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
        "Base": df_base,
        "Greedy": df_greedy,
        "Corridor": df_corridor,
    }

    # -------------------------
    # OFFLINE DP
    # -------------------------
    if RUN_OFFLINE_DP:
        df_offline_dp, summary_offline_dp = evaluate_offline_dp_case(
            df=df,
            meta=meta,
            params=BESS_PARAMS,
        )
        summary_rows.append(summary_offline_dp)
        sheets["Offline_DP"] = df_offline_dp

    summary_df = pd.DataFrame(summary_rows)

    # effect vs base
    base_loss = float(summary_base["total_loss"])
    summary_df["loss_reduction_abs"] = base_loss - summary_df["total_loss"]
    summary_df["loss_reduction_pct"] = np.where(
        base_loss != 0,
        (base_loss - summary_df["total_loss"]) / base_loss,
        np.nan,
    )

    # -------------------------
    # EXPORT
    # -------------------------
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        for sheet_name, df_sheet in sheets.items():
            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Готово. Результаты сохранены в: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()