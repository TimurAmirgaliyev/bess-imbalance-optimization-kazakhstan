from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Callable, Tuple, List

import numpy as np
import pandas as pd


# =========================================================
# bess_model.py
# ---------------------------------------------------------
# Этот модуль отвечает ТОЛЬКО за физическую модель BESS:
# - параметры батареи
# - текущее состояние батареи
# - проверку ограничений
# - обновление SOC
# - учет обязательного времени покоя после полного заряда /
#   полного разряда
# - применение действия BESS на одном шаге
# - симуляцию по массиву действий
# - симуляцию через внешний controller
#
# ВАЖНО:
# Здесь НЕ зашита "экономика" штрафов и НЕ зашит оптимизатор.
# Это сделано специально, чтобы позже можно было:
# 1) использовать простые rule-based стратегии,
# 2) подключить optimizer.py,
# 3) искать теоретически минимально возможный штраф за год.
# =========================================================


# =========================================================
# 1. ПАРАМЕТРЫ И СОСТОЯНИЕ BESS
# =========================================================

@dataclass
class BESSParams:
    """
    Параметры BESS.

    Принятое соглашение по знаку мощности:
    +p_bess_kw  -> BESS РАЗРЯЖАЕТСЯ
                   (энергия от батареи идет в сеть / в объект)
    -p_bess_kw  -> BESS ЗАРЯЖАЕТСЯ
                   (энергия забирается в батарею)

    SOC храним в долях:
    0.0 ... 1.0

    Замечание по единицам:
    ----------------------
    В твоем Excel фактические/прогнозные значения по часу могут быть
    интерпретированы как:
    - средняя мощность за час (кВт), или
    - энергия за час (кВт*ч).

    Для шага dt_h = 1.0 эти величины численно эквивалентны в формулах
    обновления SOC. Поэтому для упрощения здесь используем обозначение "_kw".
    Если потом захочешь перейти на dt != 1.0, лучше строго договориться,
    что входные значения - это средняя мощность за интервал.
    """

    energy_capacity_kwh: float              # Полная энергоемкость батареи, кВт*ч
    p_charge_max_kw: float                  # Макс. мощность заряда (по модулю), кВт
    p_discharge_max_kw: float               # Макс. мощность разряда, кВт

    soc_min: float = 0.05                   # Нижняя граница SOC
    soc_max: float = 0.95                   # Верхняя граница SOC
    soc_initial: float = 0.50               # Начальный SOC

    eta_charge: float = 0.95                # КПД заряда
    eta_discharge: float = 0.95             # КПД разряда

    self_discharge_per_hour: float = 0.0    # Саморазряд в долях/час

    # Ограничение на изменение мощности между соседними шагами.
    # Если None -> ramp-rate не учитываем.
    max_delta_p_kw_per_h: Optional[float] = None

    # Ограничения по паузе между зарядом и разрядом:
    # после достижения max SOC батарея должна стоять в нулевой мощности
    # min_rest_after_full_charge_h часов
    min_rest_after_full_charge_h: float = 0.0

    # после достижения min SOC батарея должна стоять в нулевой мощности
    # min_rest_after_full_discharge_h часов
    min_rest_after_full_discharge_h: float = 0.0

    @property
    def usable_energy_kwh(self) -> float:
        """
        Доступная (usable) энергия между soc_min и soc_max.
        """
        return (self.soc_max - self.soc_min) * self.energy_capacity_kwh

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BESSState:
    """
    Текущее состояние BESS.

    soc:
        текущий SOC в долях (0..1)

    prev_power_kw:
        мощность BESS на предыдущем шаге.
        Нужна, если позже будешь учитывать ramp-rate.

    rest_remaining_h:
        сколько часов обязательного "покоя" осталось.
        Пока > 0, BESS обязана находиться в zero-power region,
        то есть p_bess_kw = 0.

    rest_reason:
        причина текущего режима покоя:
        - "none"
        - "after_full_charge"
        - "after_full_discharge"
    """
    soc: float
    prev_power_kw: float = 0.0
    rest_remaining_h: float = 0.0
    rest_reason: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================
# 2. ВАЛИДАЦИЯ ПАРАМЕТРОВ
# =========================================================

def validate_bess_params(params: BESSParams) -> None:
    """
    Проверяет, что параметры BESS заданы корректно.
    Если что-то не так -> выбрасывает ValueError.
    """
    if params.energy_capacity_kwh <= 0:
        raise ValueError("energy_capacity_kwh должно быть > 0")

    if params.p_charge_max_kw <= 0:
        raise ValueError("p_charge_max_kw должно быть > 0")

    if params.p_discharge_max_kw <= 0:
        raise ValueError("p_discharge_max_kw должно быть > 0")

    if not (0 <= params.soc_min < params.soc_max <= 1):
        raise ValueError("Должно выполняться: 0 <= soc_min < soc_max <= 1")

    if not (params.soc_min <= params.soc_initial <= params.soc_max):
        raise ValueError("soc_initial должен быть в диапазоне [soc_min, soc_max]")

    if not (0 < params.eta_charge <= 1):
        raise ValueError("eta_charge должно быть в диапазоне (0, 1]")

    if not (0 < params.eta_discharge <= 1):
        raise ValueError("eta_discharge должно быть в диапазоне (0, 1]")

    if not (0 <= params.self_discharge_per_hour < 1):
        raise ValueError("self_discharge_per_hour должно быть в диапазоне [0, 1)")

    if params.max_delta_p_kw_per_h is not None and params.max_delta_p_kw_per_h <= 0:
        raise ValueError("max_delta_p_kw_per_h должно быть > 0 или None")

    if params.min_rest_after_full_charge_h < 0:
        raise ValueError("min_rest_after_full_charge_h должно быть >= 0")

    if params.min_rest_after_full_discharge_h < 0:
        raise ValueError("min_rest_after_full_discharge_h должно быть >= 0")


def validate_state(state: BESSState, params: BESSParams) -> None:
    """
    Проверяет корректность состояния.
    """
    if not isinstance(state.soc, (int, float, np.floating)):
        raise ValueError("state.soc должен быть числом")

    if np.isnan(state.soc):
        raise ValueError("state.soc не должен быть NaN")

    if not isinstance(state.prev_power_kw, (int, float, np.floating)):
        raise ValueError("state.prev_power_kw должен быть числом")

    if np.isnan(state.prev_power_kw):
        raise ValueError("state.prev_power_kw не должен быть NaN")

    if not isinstance(state.rest_remaining_h, (int, float, np.floating)):
        raise ValueError("state.rest_remaining_h должен быть числом")

    if np.isnan(state.rest_remaining_h):
        raise ValueError("state.rest_remaining_h не должен быть NaN")

    if state.rest_remaining_h < 0:
        raise ValueError("state.rest_remaining_h должен быть >= 0")

    if not isinstance(state.rest_reason, str):
        raise ValueError("state.rest_reason должен быть строкой")

    if state.rest_reason not in {"none", "after_full_charge", "after_full_discharge"}:
        raise ValueError("state.rest_reason имеет недопустимое значение")

    # SOC здесь пока не клипуем насильно, только проверяем на общий смысл
    if not (0 <= state.soc <= 1):
        raise ValueError("state.soc должен быть в диапазоне [0, 1]")


# =========================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ENERGY / SOC
# =========================================================

def clip_soc(soc: float, params: BESSParams) -> float:
    """
    Обрезает SOC в допустимый диапазон [soc_min, soc_max].
    """
    return min(max(float(soc), params.soc_min), params.soc_max)


def soc_to_energy_kwh(soc: float, params: BESSParams) -> float:
    """
    Переводит SOC (0..1) в запас энергии, кВт*ч.
    """
    return float(soc) * params.energy_capacity_kwh


def energy_to_soc(energy_kwh: float, params: BESSParams) -> float:
    """
    Переводит запас энергии, кВт*ч, в SOC (0..1).
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
    Создает начальное состояние BESS.
    Если soc не задан -> берем params.soc_initial.
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
# 4. ОГРАНИЧЕНИЯ ПО МОЩНОСТИ НА ТЕКУЩЕМ ШАГЕ
# =========================================================

def get_power_limits_kw(
    state: BESSState,
    params: BESSParams,
    dt_h: float = 1.0
) -> Dict[str, float]:
    """
    Возвращает допустимые границы мощности BESS на текущем шаге.

    Что учитываем:
    1) Ограничение по мощности инвертора:
       -p_charge_max_kw <= p_bess_kw <= +p_discharge_max_kw

    2) Ограничение по SOC / энергии:
       Нельзя зарядить батарею выше soc_max
       Нельзя разрядить батарею ниже soc_min

    3) Ограничение по ramp-rate (если задано):
       p_t не может слишком сильно отличаться от p_{t-1}

    4) Ограничение по обязательному времени покоя:
       Если rest_remaining_h > 0, то BESS обязана быть в zero-power region,
       то есть p_bess_kw = 0.

    Результат:
    ----------
    dict с полями:
    - p_min_kw
    - p_max_kw
    - p_min_power_only_kw
    - p_max_power_only_kw
    - p_min_energy_only_kw
    - p_max_energy_only_kw
    - p_min_ramp_kw
    - p_max_ramp_kw
    - rest_lock_active
    """
    validate_bess_params(params)
    validate_state(state, params)

    if dt_h <= 0:
        raise ValueError("dt_h должно быть > 0")

    soc = clip_soc(state.soc, params)

    # Текущий запас энергии в батарее
    e_now = soc_to_energy_kwh(soc, params)

    # Минимально и максимально допустимая энергия
    e_min = params.soc_min * params.energy_capacity_kwh
    e_max = params.soc_max * params.energy_capacity_kwh

    # -----------------------------------------------------
    # 1) Ограничения по мощности PCS / инвертора
    # -----------------------------------------------------
    p_min_power_only_kw = -params.p_charge_max_kw
    p_max_power_only_kw = params.p_discharge_max_kw

    # -----------------------------------------------------
    # 2) Ограничения по энергии / SOC
    #
    # Если мы ЗАРЯЖАЕМ BESS на |P| кВт в течение dt_h часов,
    # то внутренняя энергия увеличится на:
    #   delta_E = |P| * eta_charge * dt_h
    #
    # Значит максимально допустимый заряд по энергии:
    #   |P_charge|max <= (e_max - e_now) / (eta_charge * dt_h)
    #
    # Поскольку заряд у нас имеет ОТРИЦАТЕЛЬНЫЙ знак,
    # минимально допустимая мощность:
    #   p_min_energy_only_kw = -|P_charge|max
    # -----------------------------------------------------
    charge_limit_kw_by_energy = (e_max - e_now) / (params.eta_charge * dt_h)
    p_min_energy_only_kw = -max(0.0, charge_limit_kw_by_energy)

    # -----------------------------------------------------
    # Если мы РАЗРЯЖАЕМ BESS на P кВт в течение dt_h часов,
    # то батарея теряет внутреннюю энергию:
    #   delta_E = P / eta_discharge * dt_h
    #
    # Значит:
    #   P_discharge|max <= (e_now - e_min) * eta_discharge / dt_h
    # -----------------------------------------------------
    discharge_limit_kw_by_energy = (e_now - e_min) * params.eta_discharge / dt_h
    p_max_energy_only_kw = max(0.0, discharge_limit_kw_by_energy)

    # Пересекаем ограничения мощности и энергии
    p_min_base_kw = max(p_min_power_only_kw, p_min_energy_only_kw)
    p_max_base_kw = min(p_max_power_only_kw, p_max_energy_only_kw)

    # -----------------------------------------------------
    # 3) Ramp-rate, если задан
    # -----------------------------------------------------
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

    # -----------------------------------------------------
    # 4) Rest lock:
    # если батарея обязана отдыхать, то единственно допустимая
    # мощность = 0
    # -----------------------------------------------------
    rest_lock_active = state.rest_remaining_h > 1e-12

    if rest_lock_active:
        p_min_kw = 0.0
        p_max_kw = 0.0
    else:
        p_min_kw = p_min_after_ramp_kw
        p_max_kw = p_max_after_ramp_kw

    # На случай, если из-за ramp-rate или численных нюансов диапазон "сломался"
    # и p_min_kw > p_max_kw, приводим к безопасному значению.
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
# 5. ОБРЕЗКА КОМАНДЫ ПО ФИЗИЧЕСКИМ ОГРАНИЧЕНИЯМ
# =========================================================

def clip_power_command_kw(
        p_cmd_kw: float,
        state: BESSState,
        params: BESSParams,
        dt_h: float = 1.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Берет "желаемую" команду BESS и обрезает ее по ограничениям.

    Например:
    - controller / optimizer захотел p_cmd_kw = +7000
    - а батарея физически может сейчас только +3200
    -> тогда будет применено +3200

    Если активен обязательный отдых (rest lock), то:
    -> будет применено p_applied_kw = 0

    Возвращает:
    1) p_applied_kw  - реально примененная мощность
    2) info          - детальная диагностика ограничения
    """
    if not isinstance(p_cmd_kw, (int, float, np.floating)):
        raise ValueError("p_cmd_kw должен быть числом")

    if np.isnan(p_cmd_kw):
        raise ValueError("p_cmd_kw не должен быть NaN")

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
# 6. ОБНОВЛЕНИЕ SOC И СОСТОЯНИЯ ПОСЛЕ ПРИМЕНЕНИЯ МОЩНОСТИ
# =========================================================

def update_soc(
        state: BESSState,
        p_bess_kw: float,
        params: BESSParams,
        dt_h: float = 1.0
) -> BESSState:
    """
    Обновляет состояние BESS после применения мощности p_bess_kw.

    Соглашение:
    ----------
    +p_bess_kw -> разряд
    -p_bess_kw -> заряд

    Формулы:
    --------
    Если разряд:
        E_next = E_now - (p_bess_kw / eta_discharge) * dt_h

    Если заряд:
        E_next = E_now + (|p_bess_kw| * eta_charge) * dt_h

    Дополнительно учитывается обязательный период покоя:
    ----------------------------------------------------
    Если на текущем шаге BESS достигла:
    - soc_max при заряде
    - soc_min при разряде

    то на следующий шаг запускается таймер:
    - min_rest_after_full_charge_h
    - min_rest_after_full_discharge_h

    Важно про дискретизацию:
    ------------------------
    При dt_h = 1.0 и rest = 1.5 h реальное поведение будет консервативным,
    так как модель дискретная и следующий шаг все равно занимает целый час.
    """
    validate_bess_params(params)
    validate_state(state, params)

    if dt_h <= 0:
        raise ValueError("dt_h должно быть > 0")

    if not isinstance(p_bess_kw, (int, float, np.floating)):
        raise ValueError("p_bess_kw должен быть числом")

    if np.isnan(p_bess_kw):
        raise ValueError("p_bess_kw не должен быть NaN")

    soc_now = clip_soc(state.soc, params)
    e_now = soc_to_energy_kwh(soc_now, params)

    # Учитываем саморазряд, если он задан
    if params.self_discharge_per_hour > 0:
        e_now = e_now * max(0.0, 1.0 - params.self_discharge_per_hour * dt_h)

    # Обновляем энергию
    if p_bess_kw >= 0:
        # Разряд: батарея теряет внутреннюю энергию
        e_next = e_now - (float(p_bess_kw) / params.eta_discharge) * dt_h
    else:
        # Заряд: батарея получает энергию с учетом КПД заряда
        e_next = e_now + (abs(float(p_bess_kw)) * params.eta_charge) * dt_h

    # Жестко ограничиваем допустимым диапазоном энергии
    e_min = params.soc_min * params.energy_capacity_kwh
    e_max = params.soc_max * params.energy_capacity_kwh
    e_next = min(max(e_next, e_min), e_max)

    soc_next = energy_to_soc(e_next, params)
    soc_next = clip_soc(soc_next, params)

    # -----------------------------------------------------
    # Обновление таймера отдыха
    # -----------------------------------------------------
    # 1) Если батарея уже была в rest-режиме,
    #    то уменьшаем оставшееся время покоя на dt_h
    rest_remaining_h_after_decay = max(0.0, float(state.rest_remaining_h) - dt_h)

    if rest_remaining_h_after_decay > 1e-12:
        # Режим покоя продолжается
        next_rest_remaining_h = rest_remaining_h_after_decay
        next_rest_reason = state.rest_reason
    else:
        # Режим покоя закончился или его не было
        next_rest_remaining_h = 0.0
        next_rest_reason = "none"

        # -------------------------------------------------
        # Если на ЭТОМ шаге BESS достигла границы SOC,
        # запускаем новый mandatory rest
        #
        # Условие срабатывания:
        # - при заряде достигли soc_max
        # - при разряде достигли soc_min
        #
        # Используем np.isclose на случай численных погрешностей.
        # -------------------------------------------------
        reached_soc_max_this_step = (float(p_bess_kw) < 0) and np.isclose(soc_next, params.soc_max)
        reached_soc_min_this_step = (float(p_bess_kw) > 0) and np.isclose(soc_next, params.soc_min)

        if reached_soc_max_this_step and params.min_rest_after_full_charge_h > 0:
            next_rest_remaining_h = float(params.min_rest_after_full_charge_h)
            next_rest_reason = "after_full_charge"

        elif reached_soc_min_this_step and params.min_rest_after_full_discharge_h > 0:
            next_rest_remaining_h = float(params.min_rest_after_full_discharge_h)
            next_rest_reason = "after_full_discharge"

    next_state = BESSState(
        soc=soc_next,
        prev_power_kw=float(p_bess_kw),
        rest_remaining_h=next_rest_remaining_h,
        rest_reason=next_rest_reason
    )

    return next_state

# =========================================================
# 7. ПРИМЕНЕНИЕ ДЕЙСТВИЯ BESS К ОДНОМУ ЧАСУ
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
    Применяет одну команду BESS к одному часовому интервалу.

    Вход:
    -----
    actual_kw:
        фактическая генерация / отпуск ДО BESS

    forecast_kw:
        план / прогноз на час

    state:
        текущее состояние BESS

    p_cmd_kw:
        желаемая команда BESS (от controller или optimizer)

    Что делает:
    -----------
    1) Считает отклонение до BESS:
         deviation_before = actual - forecast

    2) Обрезает команду по физическим ограничениям:
         p_applied_kw

    3) Обновляет "фактическую" генерацию после BESS:
         actual_with_bess = actual + p_applied_kw

       Почему плюс?
       ------------
       Потому что по нашему соглашению:
       +p_bess_kw -> BESS разряжает и добавляет мощность
       -p_bess_kw -> BESS заряжает и отбирает мощность

    4) Считает новое отклонение:
         deviation_after = actual_with_bess - forecast

    5) Обновляет SOC и rest-state

    Возвращает:
    -----------
    1) result_dict - подробный словарь результатов шага
    2) next_state  - состояние BESS на следующий шаг
    """
    if not isinstance(actual_kw, (int, float, np.floating)):
        raise ValueError("actual_kw должен быть числом")
    if not isinstance(forecast_kw, (int, float, np.floating)):
        raise ValueError("forecast_kw должен быть числом")
    if np.isnan(actual_kw) or np.isnan(forecast_kw):
        raise ValueError("actual_kw и forecast_kw не должны быть NaN")

    deviation_before_kw = float(actual_kw) - float(forecast_kw)

    rest_remaining_h_start = float(state.rest_remaining_h)
    rest_reason_start = state.rest_reason
    rest_lock_active_start = rest_remaining_h_start > 1e-12

    # 1) Обрезаем команду по ограничениям
    p_applied_kw, clip_info = clip_power_command_kw(
        p_cmd_kw=p_cmd_kw,
        state=state,
        params=params,
        dt_h=dt_h
    )

    # 2) Применяем BESS к фактической генерации
    actual_with_bess_kw = float(actual_kw) + p_applied_kw
    deviation_after_kw = actual_with_bess_kw - float(forecast_kw)

    # 3) Обновляем состояние батареи
    next_state = update_soc(
        state=state,
        p_bess_kw=p_applied_kw,
        params=params,
        dt_h=dt_h
    )

    # Диагностика: достигли ли на текущем шаге предела SOC
    reached_soc_max_this_step = (p_applied_kw < 0) and np.isclose(next_state.soc, params.soc_max)
    reached_soc_min_this_step = (p_applied_kw > 0) and np.isclose(next_state.soc, params.soc_min)

    # Диагностика: запустился ли mandatory rest именно на этом переходе
    rest_started_this_step = (
            next_state.rest_remaining_h > max(0.0, rest_remaining_h_start - dt_h) + 1e-12
    )

    # Дополнительные удобные показатели
    if p_applied_kw >= 0:
        charge_power_kw = 0.0
        discharge_power_kw = p_applied_kw
    else:
        charge_power_kw = abs(p_applied_kw)
        discharge_power_kw = 0.0

    # Энергия за шаг (при dt_h = 1 это просто численно равно мощности)
    charge_energy_input_kwh = charge_power_kw * dt_h
    discharge_energy_output_kwh = discharge_power_kw * dt_h

    result = {
        #"forecast": float(forecast_kw), #убрал чтобы не дублировались колонки
        #"actual": float(actual_kw), #убрал чтобы не дублировались колонки

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
# 8. ВСПОМОГАТЕЛЬНЫЙ CONTROLLER:
#    ПЫТАЕМСЯ ПОЛНОСТЬЮ КОМПЕНСИРОВАТЬ ОТКЛОНЕНИЕ
# =========================================================

def greedy_deviation_controller(
        row: pd.Series,
        state: BESSState,
        params: BESSParams
) -> float:
    """
    Простейшая стратегия:
    пытаемся полностью компенсировать текущее отклонение.

    deviation = actual - forecast
    нужно получить deviation_after_bess = 0

    Так как:
        actual_with_bess = actual + p_bess

    Тогда:
        deviation_after_bess = actual + p_bess - forecast
                             = deviation + p_bess

    Чтобы deviation_after_bess = 0:
        p_bess = -deviation = -(actual - forecast)

    Это НЕ оптимизатор.
    Это просто базовая rule-based стратегия для тестов / сравнения.

    ВАЖНО:
    Даже если controller вернет ненулевую команду,
    физическая модель все равно обрежет ее до 0, если активен mandatory rest.
    """
    if "actual" not in row.index or "forecast" not in row.index:
        raise ValueError("Для greedy_deviation_controller нужны колонки 'actual' и 'forecast'")

    deviation_kw = float(row["actual"]) - float(row["forecast"])
    return -deviation_kw

# =========================================================
# 9. СИМУЛЯЦИЯ ПО МАССИВУ ГОТОВЫХ ДЕЙСТВИЙ
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
    Симулирует BESS по заранее заданному массиву действий actions_kw.

    Это КЛЮЧЕВАЯ функция для будущей оптимизации:
    ----------------------------------------------
    Позже optimizer.py сможет найти оптимальный вектор:
        [p_1, p_2, ..., p_8760]
    а bess_model.py просто прогонит его через физику.

    Параметры:
    ----------
    df:
        исходный DataFrame с колонками actual_col и forecast_col

    actions_kw:
        список / массив команд BESS той же длины, что и df

    initial_state:
        если None -> будет создано состояние с params.soc_initial

    Возвращает:
    -----------
    копию df с добавленными колонками BESS-симуляции
    """
    validate_bess_params(params)

    if actual_col not in df.columns:
        raise ValueError(f"В df нет колонки '{actual_col}'")
    if forecast_col not in df.columns:
        raise ValueError(f"В df нет колонки '{forecast_col}'")

    if len(actions_kw) != len(df):
        raise ValueError("Длина actions_kw должна совпадать с количеством строк df")

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

    # Добавляем результаты симуляции в исходную таблицу
    df_out = pd.concat([df_out, results_df], axis=1)

    return df_out

# =========================================================
# 10. СИМУЛЯЦИЯ ЧЕРЕЗ ВНЕШНИЙ CONTROLLER
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
    Симулирует BESS с помощью внешней функции controller.

    Это универсальный вариант:
    --------------------------
    В каждый час controller получает:
    - текущую строку df (row)
    - текущее состояние BESS (state)
    - параметры BESS (params)

    и возвращает:
    - желаемую команду мощности p_cmd_kw

    После этого модель:
    - обрежет команду по ограничениям,
    - обновит SOC,
    - учтет mandatory rest,
    - запишет результат в DataFrame.

    Преимущество:
    -------------
    Логику управления BESS можно менять БЕЗ переписывания физики модели.
    """
    validate_bess_params(params)

    if actual_col not in df.columns:
        raise ValueError(f"В df нет колонки '{actual_col}'")
    if forecast_col not in df.columns:
        raise ValueError(f"В df нет колонки '{forecast_col}'")

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

        # Временные алиасы для совместимости с controller,
        # если actual_col / forecast_col отличаются от стандартных имен
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
# 11. КРАТКАЯ СВОДКА ПО РЕЗУЛЬТАТАМ СИМУЛЯЦИИ
# =========================================================

def summarize_bess_results(df_bess: pd.DataFrame) -> Dict[str, Any]:
    """
    Возвращает краткую агрегированную сводку по результатам симуляции.

    Ожидает, что в df_bess уже есть колонки, созданные функциями симуляции:
    - p_bess_kw
    - charge_energy_input_kwh
    - discharge_energy_output_kwh
    - deviation_before
    - deviation_after_bess
    - soc_end
    - bess_was_clipped
    - bess_rest_lock_active_start
    - bess_rest_started_this_step

    Если каких-то колонок нет, часть метрик будет равна None.
    """
    summary = {}

    def safe_sum(col: str):
        return float(df_bess[col].sum()) if col in df_bess.columns else None

    def safe_last(col: str):
        return float(df_bess[col].iloc[-1]) if col in df_bess.columns and len(df_bess) > 0 else None

    summary["rows"] = int(len(df_bess))
    summary["total_charge_energy_input_kwh"] = safe_sum("charge_energy_input_kwh")
    summary["total_discharge_energy_output_kwh"] = safe_sum("discharge_energy_output_kwh")

    # Эквивалентные циклы: throughput / (2 * usable_energy)
    # throughput = charged + discharged
    # Это упрощенная инженерная оценка.
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

# внизу быстрая функция от Claude:
def apply_bess_action_fast(
    soc: float,
    rest_remaining_h: float,
    rest_reason: str,
    p_cmd_kw: float,
    # параметры батареи разворачиваем напрямую для скорости
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
    Лёгкая версия apply_bess_action для DP — без валидации и словарей.

    Возвращает:
        (actual_with_bess, next_soc, next_rest_h, next_rest_reason, throughput)
    """
    # --- clip power command ---
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

    # --- update SOC ---
    e_now = soc * energy_capacity_kwh
    if p_applied >= 0.0:
        e_next = e_now - (p_applied / eta_discharge) * dt_h
    else:
        e_next = e_now + (-p_applied * eta_charge) * dt_h

    e_min = soc_min * energy_capacity_kwh
    e_max = soc_max * energy_capacity_kwh
    e_next = min(max(e_next, e_min), e_max)
    soc_next = min(max(e_next / energy_capacity_kwh, soc_min), soc_max)

    # --- update rest timer ---
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

    # throughput для degradation_weight
    throughput = abs(p_applied) * dt_h

    return p_applied, soc_next, next_rest_h, next_rest_reason, throughput

# =========================================================
# 12. БЫСТРЫЙ ТЕСТ МОДУЛЯ
# =========================================================

if __name__ == "__main__":
    from io_data import load_input_data

    # Путь к твоему Excel
    file_path = "import/korem.xlsx"

    # Загружаем данные через твой уже существующий модуль
    df, meta = load_input_data(file_path)

    # Пример параметров BESS
    params = BESSParams(
        energy_capacity_kwh=10000,  # 10 МВт*ч
        p_charge_max_kw=5000,  # 5 МВт заряд
        p_discharge_max_kw=5000,  # 5 МВт разряд
        soc_min=0.10,
        soc_max=0.90,
        soc_initial=0.50,
        eta_charge=0.95,
        eta_discharge=0.95,
        self_discharge_per_hour=0.0,
        max_delta_p_kw_per_h=None,

        # Ограничения из картинки:
        min_rest_after_full_charge_h=1.5,  # 90 мин
        min_rest_after_full_discharge_h=1.5  # 90 мин
    )

    # Базовый тест: симуляция по простому greedy-controller
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
        "datetime",
        "forecast",
        "actual",
        "deviation_before",
        "p_cmd_kw",
        "p_bess_kw",
        "actual_with_bess",
        "deviation_after_bess",
        "soc_start",
        "soc_end",
        "bess_rest_lock_active_start",
        "bess_rest_remaining_h_start",
        "bess_rest_remaining_h_end",
        "bess_rest_started_this_step",
        "bess_reached_soc_max_this_step",
        "bess_reached_soc_min_this_step",
        "bess_was_clipped"
    ]

    existing_cols_to_show = [c for c in cols_to_show if c in df_bess.columns]
    print(df_bess[existing_cols_to_show].head(20))

    print("\n=== SUMMARY ===")
    summary = summarize_bess_results(df_bess)
    for k, v in summary.items():
        print(f"{k}: {v}")


#экспорт в Excel:

from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#df_bess.to_excel(f"export/output_after_bess_{timestamp}.xlsx", index=False, engine="openpyxl")

