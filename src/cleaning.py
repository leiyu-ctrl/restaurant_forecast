"""Очистка данных и обработка пропусков."""

import pandas as pd

def add_missing_dates(data: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Восстановить календарь и добавить информацию о праздниках

    calendar — таблица дат с is_holiday (например, вывод load_calendar),
    её передаём отдельно, чтобы праздник был известен и для пропущенных дней.
    """
    data = data.copy()

    # активный период каждого ресторана — от его первой до последней
    # реальной записи.
    span = data.groupby("restaurant_id")["date"].agg(first="min", last="max")

    date_range = pd.date_range(data["date"].min(), data["date"].max())
    full_index = pd.MultiIndex.from_product(
        [data["restaurant_id"].unique(), date_range],
        names=["restaurant_id", "date"],
    )
    full_calendar = pd.DataFrame(index=full_index).reset_index()
    full_calendar = full_calendar.merge(span, on="restaurant_id", how="left")
    full_calendar = full_calendar[
        (full_calendar["date"] >= full_calendar["first"])
        & (full_calendar["date"] <= full_calendar["last"])
    ].drop(columns=["first", "last"])

    result = full_calendar.merge(data, on=["restaurant_id", "date"], how="left")
    result = result.merge(calendar[["date", "is_holiday"]], on="date", how="left", suffixes=("", "_cal"))
    result["genre"] = result.groupby("restaurant_id")["genre"].transform("first")
    result["area"] = result.groupby("restaurant_id")["area"].transform("first")
    result["is_holiday"] = result["is_holiday_cal"]
    return result.drop(columns="is_holiday_cal")


def fill_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    """Добавить календарные признаки."""
    data = data.copy()
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["is_weekend"] = data["day_of_week"].isin([5, 6])
    return data


def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Разметить пропуски: 1 — регулярный выходной, 2 — праздник, 3 — остальное."""
    data = data.copy()
    data["is_missing"] = data["guests"].isna()

    # доля пропусков по (ресторан, день недели) -> у каких пар это регулярный выходной
    off_share = data.groupby(["restaurant_id", "day_of_week"])["is_missing"].transform("mean")
    data["is_regular_off"] = off_share >= 0.8

    data["missing_type"] = 0
    data.loc[data["is_missing"] & data["is_regular_off"], "missing_type"] = 1
    data.loc[data["is_missing"] & ~data["is_regular_off"] & data["is_holiday"].eq(1), "missing_type"] = 2
    data.loc[data["is_missing"] & data["missing_type"].eq(0), "missing_type"] = 3

    data.loc[data["missing_type"] == 1, "guests"] = 0

    # длина серии подряд идущих пропусков типа 3 (остальные, причина неизвестна)
    is_type3 = data["missing_type"].eq(3)
    run_id = (~is_type3).groupby(data["restaurant_id"]).cumsum()
    run_len = is_type3.groupby([data["restaurant_id"], run_id]).transform("sum")
    data["is_long_gap"] = is_type3 & (run_len >= 3)

    return data


def clean_data(data: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Подготовить данные: восстановить календарь и разметить пропуски."""
    data = add_missing_dates(data, calendar)
    data = fill_calendar_features(data)
    data = handle_missing_values(data)
    return data