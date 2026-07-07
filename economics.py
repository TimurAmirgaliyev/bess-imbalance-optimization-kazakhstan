
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from io_data import load_input_data


def calculate_balancing_penalty(
    df: pd.DataFrame,
    meta: Dict[str, Any],
    actual_col: str = "actual",
    forecast_col: str = "forecast",
    prefix: str = "py_",
) -> pd.DataFrame:
    """
    Считает экономику / штраф на балансирующем рынке
    для произвольной колонки факта.

    Параметры
    ---------
    df : pd.DataFrame
        Исходный DataFrame с forecast и actual/actual_with_bess.
    meta : dict
        Словарь параметров (tariff, acceptable_range_plus/minus,
        increasing_factor, decreasing_factor и т.д.).
    actual_col : str
        Какая колонка считается "фактом" для расчета:
        - 'actual'            -> сценарий без BESS
        - 'actual_with_bess'  -> сценарий с BESS
    forecast_col : str
        Имя колонки прогноза.
    prefix : str
        Префикс выходных расчетных колонок:
        - 'base_'  -> без BESS
        - 'bess_'  -> с BESS
        - 's1_'    -> сценарий 1 и т.д.

    Возвращает
    ----------
    pd.DataFrame
        Копию df с добавленными расчетными колонками.
    """
    df = df.copy()

    if forecast_col not in df.columns:
        raise KeyError(f"Колонка forecast_col='{forecast_col}' не найдена в DataFrame")
    if actual_col not in df.columns:
        raise KeyError(f"Колонка actual_col='{actual_col}' не найдена в DataFrame")

    required_meta = [
        "tariff",
        "acceptable_range_plus",
        "acceptable_range_minus",
        "decreasing_factor",
        "increasing_factor",
    ]
    missing_meta = [k for k in required_meta if k not in meta or meta[k] is None]
    if missing_meta:
        raise KeyError(f"В meta отсутствуют обязательные ключи: {missing_meta}")

    # короткие алиасы для читаемости
    fact = df[actual_col]
    forecast = df[forecast_col]
    tariff = meta["tariff"]
    acceptable_range_plus = meta["acceptable_range_plus"]
    acceptable_range_minus = meta["acceptable_range_minus"]
    decreasing_factor = meta["decreasing_factor"]
    increasing_factor = meta["increasing_factor"]

    # имена колонок с префиксом
    c_deviation = f"{prefix}deviation"
    c_deviation_pct = f"{prefix}deviation_pct"
    c_sales_forecast = f"{prefix}sales_forecast"
    c_positive_dev_pct = f"{prefix}positive_dev_pct"
    c_within_5_pct_positive = f"{prefix}within_5_pct_positive"
    c_sales_within_5pct = f"{prefix}sales_within_5pct"
    c_beyond_5_pct_positive = f"{prefix}beyond_5_pct_positive"
    c_sales_beyond_5pct = f"{prefix}sales_beyond_5pct"
    c_negative_dev_pct = f"{prefix}negative_dev_pct"
    c_within_5_pct_negative = f"{prefix}within_5_pct_negative"
    c_purchase_within_5pct = f"{prefix}purchase_within_5pct"
    c_beyond_5_pct_negative = f"{prefix}beyond_5_pct_negative"
    c_purchase_beyond_5pct = f"{prefix}purchase_beyond_5pct"
    c_total_sales_penalized = f"{prefix}total_sales_penalized"
    c_unpenalized_sales = f"{prefix}unpenalized_sales"
    c_loss = f"{prefix}loss"

    """
    Собственный расчет штрафа в Python
    """
    df[c_deviation] = fact - forecast

    df[c_deviation_pct] = np.where(
        forecast > 0,
        df[c_deviation] / forecast,
        np.where(fact != 0, 1.0, 0.0)
    )

    df[c_sales_forecast] = forecast * tariff

    """
    Отрицательный дисбаланс:
    (в твоих текущих комментариях так назван блок positive deviation)
    """

    # 1. Позитивная часть отклонения
    df[c_positive_dev_pct] = df[c_deviation_pct].clip(lower=0)

    # 2. Часть в пределах лимита
    df[c_within_5_pct_positive] = df[c_positive_dev_pct].clip(
        upper=acceptable_range_plus
    )

    # 3. Финальный расчет
    df[c_sales_within_5pct] = np.where(
        forecast > 0,
        df[c_within_5_pct_positive] * forecast * tariff,
        acceptable_range_plus * fact * tariff
    )

    # Часть за пределами лимита
    df[c_beyond_5_pct_positive] = (
        df[c_positive_dev_pct] - acceptable_range_plus
    ).clip(lower=0)

    # Финальный расчет
    df[c_sales_beyond_5pct] = np.where(
        forecast > 0,
        df[c_beyond_5_pct_positive] * forecast * tariff * decreasing_factor,
        df[c_beyond_5_pct_positive] * fact * tariff * decreasing_factor
    )

    """
    Положительный дисбаланс:
    (в твоих текущих комментариях так назван блок negative deviation)
    """

    # 1. Отрицательная часть отклонения
    df[c_negative_dev_pct] = df[c_deviation_pct].clip(upper=0)

    # 2. Часть в пределах лимита
    df[c_within_5_pct_negative] = df[c_negative_dev_pct].clip(
        lower=acceptable_range_minus
    )

    # 3. Финальный расчет
    df[c_purchase_within_5pct] = df[c_within_5_pct_negative] * forecast * tariff

    # Часть за пределами лимита
    df[c_beyond_5_pct_negative] = (
        df[c_negative_dev_pct] - acceptable_range_minus
    ).clip(upper=0)

    # Финальный расчет
    df[c_purchase_beyond_5pct] = np.where(
        forecast > 0,
        df[c_beyond_5_pct_negative] * forecast * tariff * increasing_factor,
        df[c_beyond_5_pct_negative] * fact * tariff * increasing_factor
    )

    """
    Итоговая продажа электроэнергии с учетом штрафов
    Total Sales Penalized =
        Sales forecast +
        Sales within 5% +
        Sales beyond 5% +
        Purchase within 5% +
        Purchase beyond 5%
    """
    df[c_total_sales_penalized] = (
        df[c_sales_forecast]
        + df[c_sales_within_5pct]
        + df[c_sales_beyond_5pct]
        + df[c_purchase_within_5pct]
        + df[c_purchase_beyond_5pct]
    )

    # Доп. полезные итоговые колонки
    df[c_unpenalized_sales] = fact * tariff
    df[c_loss] = df[c_unpenalized_sales] - df[c_total_sales_penalized]

    return df


def summarize_penalty(df: pd.DataFrame, prefix: str = "py_") -> Dict[str, float]:
    """
    Возвращает агрегированные суммы по рассчитанным колонкам.
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

    out = {}
    for col in cols:
        out[col] = float(df[col].sum()) if col in df.columns else np.nan
    return out


if __name__ == "__main__":
    # Быстрый тест как раньше, чтобы модуль можно было запускать отдельно
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
