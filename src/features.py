"""Признаки для прогноза guests.

Все признаки должны быть доступны на момент прогноза.
Лаги используют только прошлые значения guests.
Скользящие статистики считаются со сдвигом на 1 день,
чтобы текущий target не попал в признаки.

Порядок подготовки данных:
    load_data -> clean_data -> build_features

Служебные признаки(is_missing, missing_type,
is_long_gap, is_regular_off) сюда не идут — см. FEATURE_COLUMNS и комментарии
в конце файла.
"""

from collections.abc import Sequence
import pandas as pd

DEFAULT_LAGS: tuple[int, ...] = (1, 7, 14, 21, 28)
DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (7, 28)
DEFAULT_HOLIDAY_WINDOW: int = 3


def add_lag_features(
    data: pd.DataFrame,
    lags: Sequence[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Лаги guests по ресторану. Данные должны быть отсортированы по (restaurant_id, date)."""
    data = data.sort_values(["restaurant_id", "date"]).copy()
    grouped_guests = data.groupby("restaurant_id")["guests"]
    for lag in lags:
        data[f"guests_lag_{lag}"] = grouped_guests.shift(lag)
    return data


def add_rolling_features(
    data: pd.DataFrame,
    windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Скользящие среднее/std по guests, посчитанные ТОЛЬКО по дням до текущего
    (shift(1) перед rolling), иначе rolling включает сам таргет текущего дня.
    """
    data = data.copy()
    shifted = data.groupby("restaurant_id")["guests"].shift(1)
    for window in windows:
        data[f"guests_roll_mean_{window}"] = shifted.groupby(data["restaurant_id"]).transform(
            lambda s: s.rolling(window, min_periods=max(3, window // 2)).mean()
        )
        data[f"guests_roll_std_{window}"] = shifted.groupby(data["restaurant_id"]).transform(
            lambda s: s.rolling(window, min_periods=max(3, window // 2)).std()
        )
    return data


def add_holiday_phase(data: pd.DataFrame, calendar: pd.DataFrame, window: int = DEFAULT_HOLIDAY_WINDOW) -> pd.DataFrame:
    """Фаза дня относительно ближайшего праздника: regular / holiday / before_k / after_k.

    Считается один раз по календарю и мёрджится по date 
    """
    data = data.copy()
    cal = (
        calendar[["date", "is_holiday"]]
        .drop_duplicates("date")
        .set_index("date")
        .sort_index()
        .asfreq("D")
    )
    h = cal["is_holiday"].fillna(0)

    phase = pd.Series("regular", index=cal.index, name="holiday_phase")
    phase[h == 1] = "holiday"
    for k in range(1, window + 1):
        phase[(h.shift(k) == 1) & (phase == "regular")] = f"after_{k}"
        phase[(h.shift(-k) == 1) & (phase == "regular")] = f"before_{k}"

    data = data.merge(phase, left_on="date", right_index=True, how="left")
    data["holiday_phase"] = data["holiday_phase"].fillna("regular").astype("category")
    return data


def build_features(
    data: pd.DataFrame,
    calendar: pd.DataFrame,
    lags: Sequence[int] = DEFAULT_LAGS,
    rolling_windows: Sequence[int] = DEFAULT_ROLLING_WINDOWS,
    holiday_window: int = DEFAULT_HOLIDAY_WINDOW,
) -> pd.DataFrame:
    """Собирает все признаки поверх результата cleaning.clean_data.

    Порядок вызовов: load_data -> clean_data -> build_features.
    (build_features не трогает revenue/add_synthetic_revenue — она для
    EDA/отчётности, см. data.py)
    """
    data = data.sort_values(["restaurant_id", "date"]).copy()
    data = add_lag_features(data, lags)
    data = add_rolling_features(data, rolling_windows)
    data = add_holiday_phase(data, calendar, holiday_window)
    return data


# Колонки, которые отдаём в модель как признаки.
#   guests        - таргет
#   revenue       - детерминированная функция guests (утечка), см. data.py
#   is_missing, missing_type, is_long_gap, is_regular_off - построены из
#   факта пропуска guests в этот же день, для будущего дня неизвестны
FEATURE_COLUMNS = [
    "restaurant_id",
    "genre",
    "day_of_week",
    "is_weekend",
    "month",
    "is_holiday",
    "holiday_phase",
] + [f"guests_lag_{lag}" for lag in DEFAULT_LAGS] + [
    f"guests_roll_{stat}_{window}"
    for window in DEFAULT_ROLLING_WINDOWS
    for stat in ("mean", "std")
]