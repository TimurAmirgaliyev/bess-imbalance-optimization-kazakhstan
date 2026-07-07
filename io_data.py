import pandas as pd
from typing import Tuple, Dict, Any


# -----------------------------
# 1. Настройки: имя листа и маппинг колонок
# -----------------------------

DEFAULT_SHEET_NAME = "Лист1"

COLUMN_RENAME_MAP = {
    "Date and time": "datetime",
    "Forecast": "forecast",
    "Actual": "actual",
    "Deviation": "deviation",
    "Deviation percentage": "deviation_pct",
    "Sales forecast": "sales_forecast",
    "Sales within 5%": "sales_within_5pct",
    "Sales beyond 5%": "sales_beyond_5pct",
    "Purchase within 5%": "purchase_within_5pct",
    "Purchase beyond 5%": "purchase_beyond_5pct",
    "Total Sales Penalized": "total_sales_penalized",
    "Unpenalized Sales": "unpenalized_sales",
    "Loss": "loss",
    "ABS Deviation": "abs_deviation",
    "ABS Deviation percentage": "abs_deviation_pct",
    "Forecast + Actual": "forecast_plus_actual",
    "(Actual - Forecast)^2": "sq_error",
    "Tariff": "tariff",
    "Increasing factor": "increasing_factor",
    "Decreasing factor": "decreasing_factor",
    "Acceptable range (+)": "acceptable_range_plus",
    "Acceptable range (-)": "acceptable_range_minus",
    "P installed AC, kW": "p_installed_ac_kw",
    "MAPE (forecast based)": "mape_forecast_based",
    "MAE": "mae",
    "NMAE": "nmae",
    "RMSE": "rmse",
    "nRMSE": "nrmse",
}

# Колонки, которые считаем "основным рядом"
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

# Колонки, которые считаем "метаданными / константами"
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


# -----------------------------
# 2. Вспомогательные функции
# -----------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Переименовывает известные колонки и удаляет полностью пустые/Unnamed-колонки.
    """
    df = df.copy()

    # Удаляем пустые технические колонки вроде Unnamed: 13 / Unnamed: 18 / Unnamed: 25
    drop_cols = [col for col in df.columns if str(col).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Переименование в удобные snake_case имена
    df = df.rename(columns=COLUMN_RENAME_MAP)

    return df


def _convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит datetime и числовые колонки к правильным типам.
    """
    df = df.copy()

    # datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # все остальные столбцы (кроме datetime) пытаемся сделать numeric
    for col in df.columns:
        if col != "datetime":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def extract_meta(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Извлекает разовые параметры и KPI из первых непустых значений в соответствующих колонках.
    """
    meta = {}

    for col in META_COLUMNS:
        if col in df.columns:
            non_null = df[col].dropna()
            meta[col] = non_null.iloc[0] if not non_null.empty else None

    return meta


def validate_required_columns(df: pd.DataFrame, required_columns=None) -> None:
    """
    Проверяет, что обязательные колонки существуют.
    """
    if required_columns is None:
        required_columns = ["datetime", "forecast", "actual"]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"В Excel не найдены обязательные колонки: {missing}")


def check_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """
    Возвращает словарь с количеством пропусков по колонкам.
    """
    return df.isna().sum().to_dict()


def prepare_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Готовит основной временной ряд:
    - оставляет только нужные колонки,
    - сортирует по datetime,
    - удаляет строки без даты,
    - удаляет дубликаты datetime,
    - сбрасывает индекс.
    """
    df = df.copy()

    existing_core_cols = [col for col in CORE_COLUMNS if col in df.columns]
    keep_cols = ["datetime"] + [col for col in existing_core_cols if col != "datetime"]

    df = df[keep_cols]

    # удалить строки без даты
    df = df[df["datetime"].notna()].copy()

    # сортировка по времени
    df = df.sort_values("datetime")

    # убрать дубликаты дат, если вдруг есть
    df = df.drop_duplicates(subset="datetime", keep="first")

    # reset index
    df = df.reset_index(drop=True)

    return df


# -----------------------------
# 3. Основная функция загрузки
# -----------------------------

def load_input_data(
    path: str,
    sheet_name: str = DEFAULT_SHEET_NAME
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Читает ваш Excel, чистит данные и возвращает:
    1) df_main  - основной часовой ряд
    2) meta     - словарь с тарифом, коэффициентами, диапазоном, установленной мощностью и KPI
    """
    # Чтение Excel
    df_raw = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")

    # Нормализация названий колонок
    df_raw = _normalize_columns(df_raw)

    # Приведение типов
    df_raw = _convert_types(df_raw)

    # Проверка обязательных полей
    validate_required_columns(df_raw, required_columns=["datetime", "forecast", "actual"])

    # Извлечение метаданных / констант
    meta = extract_meta(df_raw)

    # Подготовка основного ряда
    df_main = prepare_timeseries(df_raw)

    # Пропуски в основном ряду
    missing_info = check_missing_values(df_main)
    meta["missing_values_main_table"] = missing_info

    return df_main, meta


# -----------------------------
# 4. Быстрый тест файла
# -----------------------------

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