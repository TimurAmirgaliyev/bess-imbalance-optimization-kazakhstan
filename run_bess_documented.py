"""
run_bess.py
===========

Main simulation runner for BESS scenario comparison.

Purpose
-------
This script orchestrates the full analysis pipeline: loading input data,
computing the baseline penalty without a BESS, running physical BESS
simulations for one or more configured scenarios, evaluating economics
for each scenario, comparing results against the baseline, and exporting
everything to a structured Excel workbook.

It is the top-level entry point intended to be run directly.  All
physical modelling, economic calculations, and I/O logic live in the
modules it imports.

Pipeline Overview
-----------------
1. Load the source Excel file via io_data.load_input_data().
2. Compute baseline (no-BESS) balancing penalties via
   economics.calculate_balancing_penalty().
3. For each BESS scenario:
   a. Run a greedy physical simulation via
      bess_model.simulate_with_controller().
   b. Compute post-BESS balancing penalties via
      economics.calculate_balancing_penalty().
   c. Collect physical and economic summary metrics.
4. Build a cross-scenario comparison table and compute the economic
   benefit of each BESS scenario relative to the baseline.
5. Export per-step hourly results and the summary table to Excel.

Output File
-----------
A timestamped Excel workbook written to the ``export/`` directory with
the following sheets:

- ``Summary``   — one row per scenario with physical and economic KPIs
- ``Base``      — hourly detail for the no-BESS baseline
- ``Bess_1``    — hourly detail for scenario bess_1
- ``Bess_2``    — hourly detail for scenario bess_2

Scope — Intentional Exclusions
-------------------------------
This script contains no physical modelling, penalty calculation logic,
or optimisation.  Those concerns belong in their respective modules:

- Physical battery model  → bess_model.py
- Economic penalties      → economics.py
- Input data loading      → io_data.py
- MILP optimisation       → optimizer_milp.py

Author
------
Timur Amirgaliyev

Last Updated
------------
2026-06-10
"""

import pandas as pd

from io_data import load_input_data
from economics import calculate_balancing_penalty, summarize_penalty
from bess_model import (
    BESSParams,
    simulate_with_controller,
    greedy_deviation_controller,
    summarize_bess_results,
)

from datetime import datetime


# =========================================================
# 1. CONFIGURATION
# =========================================================

timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"export/run_bess_results_{timestamp}.xlsx"

INPUT_FILE  = "import/korem.xlsx"
SHEET_NAME  = "Лист1"

# Columns always included in the hourly export sheets.
# Additional scenario-specific columns are appended automatically.
BASE_EXPORT_COLS = [
    "datetime",
    "forecast",
    "actual",
]


# =========================================================
# 2. LOAD INPUT DATA
# =========================================================

df, meta = load_input_data(INPUT_FILE, sheet_name=SHEET_NAME)


# =========================================================
# 3. BASELINE SCENARIO (no BESS)
# =========================================================

# Compute per-step balancing penalties using the raw actual generation.
# All output columns are prefixed with "base_" to avoid name collisions
# when merging with BESS scenario results.
df_base = calculate_balancing_penalty(
    df=df,
    meta=meta,
    actual_col="actual",
    forecast_col="forecast",
    prefix="base_",
)

base_summary = summarize_penalty(df_base, prefix="base_")

base_summary_row = {
    "scenario":             "no_bess",
    "energy_capacity_kwh":  0.0,
    "p_charge_max_kw":      0.0,
    "p_discharge_max_kw":   0.0,
    **base_summary,
}


# =========================================================
# 4. BESS SCENARIOS
# =========================================================

# Each entry defines a scenario name and a fully configured BESSParams.
# Add or remove entries here to change which scenarios are evaluated.
scenarios = [
    {
        "scenario": "bess_1",
        "params": BESSParams(
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
        ),
    },
    {
        "scenario": "bess_2",
        "params": BESSParams(
            energy_capacity_kwh=60_000,
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
        ),
    },
]

summary_rows  = [base_summary_row]
hourly_results = {"base_hourly": df_base}

for s in scenarios:
    scenario_name = s["scenario"]
    params        = s["params"]

    # ------------------------------------------------------------------
    # Step a: physical BESS simulation
    #
    # The greedy controller attempts to fully compensate the deviation
    # at every step.  The physical model clips commands to the feasible
    # interval (SOC limits, power limits, rest periods).
    # ------------------------------------------------------------------
    df_bess = simulate_with_controller(
        df=df,
        controller=greedy_deviation_controller,
        params=params,
        dt_h=1.0,
        initial_state=None,
        actual_col="actual",
        forecast_col="forecast",
    )

    actual_bess_col = "actual_with_bess"

    if actual_bess_col not in df_bess.columns:
        raise KeyError(
            f"Column '{actual_bess_col}' not found after simulate_with_controller(). "
            f"Available columns: {df_bess.columns.tolist()}"
        )

    # ------------------------------------------------------------------
    # Step b: economics on post-BESS actual generation
    #
    # Column prefix is set to the scenario name so that results from
    # different scenarios can coexist in the same DataFrame.
    # ------------------------------------------------------------------
    df_bess_calc = calculate_balancing_penalty(
        df=df_bess,
        meta=meta,
        actual_col=actual_bess_col,
        forecast_col="forecast",
        prefix=f"{scenario_name}_",
    )

    # ------------------------------------------------------------------
    # Step c: collect summary metrics
    # ------------------------------------------------------------------
    penalty_summary = summarize_penalty(df_bess_calc, prefix=f"{scenario_name}_")

    try:
        bess_summary = summarize_bess_results(df_bess)
    except Exception:
        bess_summary = {}

    summary_row = {
        "scenario":             scenario_name,
        "energy_capacity_kwh":  params.energy_capacity_kwh,
        "p_charge_max_kw":      params.p_charge_max_kw,
        "p_discharge_max_kw":   params.p_discharge_max_kw,
        "soc_min":              params.soc_min,
        "soc_max":              params.soc_max,
        "soc_initial":          params.soc_initial,
        "eta_charge":           params.eta_charge,
        "eta_discharge":        params.eta_discharge,
        **penalty_summary,
        **bess_summary,
    }

    summary_rows.append(summary_row)
    hourly_results[f"{scenario_name}_hourly"] = df_bess_calc


# =========================================================
# 5. CROSS-SCENARIO COMPARISON
# =========================================================

summary_df = pd.DataFrame(summary_rows)

# Compute the economic benefit of each BESS scenario relative to the
# no-BESS baseline, measured as the increase in total penalised sales.
#
#   benefit = scenario_total_sales_penalized - base_total_sales_penalized
#
# A positive value means the BESS scenario recovered more revenue than
# the baseline after penalties.
if "base_total_sales_penalized" in summary_df.columns:
    base_total = summary_df.loc[
        summary_df["scenario"] == "no_bess",
        "base_total_sales_penalized"
    ].iloc[0]

    penalized_cols = [
        c for c in summary_df.columns
        if c.endswith("_total_sales_penalized") and c != "base_total_sales_penalized"
    ]

    for col in penalized_cols:
        effect_col = col.replace("_total_sales_penalized", "_benefit_vs_base")
        summary_df[effect_col] = summary_df[col] - base_total


# =========================================================
# 6. EXPORT TO EXCEL
# =========================================================

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    df_base.to_excel(writer, sheet_name="Base", index=False)
    hourly_results["bess_1_hourly"].to_excel(writer, sheet_name="Bess_1", index=False)
    hourly_results["bess_2_hourly"].to_excel(writer, sheet_name="Bess_2", index=False)

print(f"Done. Results saved to: {OUTPUT_FILE}")