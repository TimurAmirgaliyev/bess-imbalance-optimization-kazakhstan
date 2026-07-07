import pandas as pd

from io_data import load_input_data
from economics import calculate_balancing_penalty, summarize_penalty
from bess_model import (
    BESSParams,
    simulate_with_controller,
    greedy_deviation_controller,
    summarize_bess_results,
)


# =========================================================
# 1. Настройки
# =========================================================

from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"export/run_bess_results_{timestamp}.xlsx"

INPUT_FILE = "import/korem.xlsx"
SHEET_NAME = "Лист1"

# Какие колонки сохранять в почасовой детализации
BASE_EXPORT_COLS = [
    "datetime",
    "forecast",
    "actual",
]

# =========================================================
# 2. Загрузка исходных данных
# =========================================================
df, meta = load_input_data(INPUT_FILE, sheet_name=SHEET_NAME)

# =========================================================
# 3. Сценарий БЕЗ BESS
# =========================================================
df_base = calculate_balancing_penalty(
    df=df,
    meta=meta,
    actual_col="actual",
    forecast_col="forecast",
    prefix="base_",
)

base_summary = summarize_penalty(df_base, prefix="base_")
base_summary_row = {
    "scenario": "no_bess",
    "energy_capacity_kwh": 0.0,
    "p_charge_max_kw": 0.0,
    "p_discharge_max_kw": 0.0,
    **base_summary,
}

# =========================================================
# 4. Сценарии С BESS
# =========================================================
scenarios = [
    {
        "scenario": "bess_1",
        "params": BESSParams(
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
        ),
    },
    {
        "scenario": "bess_2",
        "params": BESSParams(
            energy_capacity_kwh=60000,
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
        ),
    },
]

summary_rows = [base_summary_row]
hourly_results = {
    "base_hourly": df_base
}

for s in scenarios:
    scenario_name = s["scenario"]
    params = s["params"]

    # 1) Физическая симуляция BESS
    df_bess = simulate_with_controller(
        df=df,
        controller=greedy_deviation_controller,
        params=params,
        dt_h=1.0,
        initial_state=None,
        actual_col="actual",
        forecast_col="forecast",
    )

    # Удаляем дублирующиеся колонки, оставляя первое вхождение
    #df_bess = df_bess.loc[:, ~df_bess.columns.duplicated()].copy()

    # !!! ВАЖНО:
    # ниже предполагается, что в результате есть колонка actual_with_bess

    actual_bess_col = "actual_with_bess"

    if actual_bess_col not in df_bess.columns:
        raise KeyError(
            f"Колонка '{actual_bess_col}' не найдена после simulate_with_controller(). "
            f"Проверь реальное имя через print(df_bess.columns.tolist())"
        )

    # 2) Экономика после BESS
    df_bess_calc = calculate_balancing_penalty(
        df=df_bess,
        meta=meta,
        actual_col=actual_bess_col,
        forecast_col="forecast",
        prefix=f"{scenario_name}_",
    )

    # 3) Сводка по штрафам
    penalty_summary = summarize_penalty(df_bess_calc, prefix=f"{scenario_name}_")

    # 4) Сводка по физике BESS
    try:
        bess_summary = summarize_bess_results(df_bess)
    except Exception:
        bess_summary = {}

    summary_row = {
        "scenario": scenario_name,
        "energy_capacity_kwh": params.energy_capacity_kwh,
        "p_charge_max_kw": params.p_charge_max_kw,
        "p_discharge_max_kw": params.p_discharge_max_kw,
        "soc_min": params.soc_min,
        "soc_max": params.soc_max,
        "soc_initial": params.soc_initial,
        "eta_charge": params.eta_charge,
        "eta_discharge": params.eta_discharge,
        **penalty_summary,
        **bess_summary,
    }

    summary_rows.append(summary_row)
    hourly_results[f"{scenario_name}_hourly"] = df_bess_calc

# =========================================================
# 5. Сравнение сценариев
# =========================================================
summary_df = pd.DataFrame(summary_rows)

# если есть база, посчитаем экономический эффект vs no_bess
if "base_total_sales_penalized" in summary_df.columns:
    base_total = summary_df.loc[
        summary_df["scenario"] == "no_bess",
        "base_total_sales_penalized"
    ].iloc[0]

    # найдём все колонки *_total_sales_penalized кроме base_
    penalized_cols = [
        c for c in summary_df.columns
        if c.endswith("_total_sales_penalized") and c != "base_total_sales_penalized"
    ]

    for col in penalized_cols:
        effect_col = col.replace("_total_sales_penalized", "_benefit_vs_base")
        summary_df[effect_col] = summary_df[col] - base_total

# =========================================================
# 6. Экспорт в Excel
# =========================================================

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    df_base.to_excel(writer, sheet_name="Base", index=False)
    hourly_results["bess_1_hourly"].to_excel(writer, sheet_name="Bess_1", index=False)
    hourly_results["bess_2_hourly"].to_excel(writer, sheet_name="Bess_2", index=False)

print(f"Готово. Результаты сохранены в: {OUTPUT_FILE}")