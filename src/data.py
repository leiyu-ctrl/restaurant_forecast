"""Dataloader на основе датасета Kaggle Recruit Restaurant Visitor Forecasting.

Сырые файлы с каггла положены в data/raw/:
    air_visit_data.csv  - посетители по ресторанам и дням (только дни с записью)
    air_store_info.csv  - жанр и район ресторана
    date_info.csv       - метки дней праздников Японии

Выручки и среднего чека в исходном датасете нет, поэтому данные о выручке синтетические 
(функция add_synthetic_revenue)
"""
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42

def load_data(raw_dir: str | Path) -> pd.DataFrame:
    """Загружает исходные таблицы и объединяет их в один df."""
    raw_dir = Path(raw_dir)
    visits = pd.read_csv(raw_dir / "air_visit_data.csv", parse_dates=["visit_date"])
    stores = pd.read_csv(raw_dir / "air_store_info.csv")
    calendar = pd.read_csv(raw_dir / "date_info.csv", parse_dates=["calendar_date"])

    visits = visits.rename(
        columns={
            "air_store_id": "restaurant_id",
            "visit_date": "date",
            "visitors": "guests",
        }
    )
    stores = stores.rename(
        columns={
            "air_store_id": "restaurant_id",
            "air_genre_name": "genre",
            "air_area_name": "area",
        }
    )
    calendar = calendar.rename(
        columns={
            "calendar_date": "date",
            "holiday_flg": "is_holiday",
        }
    )
    data = visits.merge(calendar[["date", "is_holiday"]], on="date", how="left")
    data = data.merge(stores[["restaurant_id", "genre", "area"]], on="restaurant_id", how="left")

    return data.sort_values(["restaurant_id", "date"]).reset_index(drop=True)


def add_synthetic_revenue(
    data: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Добавляет синтетическую выручку: гости * средний чек ресторана * шум.
    В исходных данных выручки нет, в модели не используется. 
    """
    data = data.copy()
    rng = np.random.default_rng(random_state)
    restaurants = data["restaurant_id"].unique()

    # у каждого ресторана свой средний чек (в йенах)
    average_checks = pd.Series(
        rng.normal(loc=2400, scale=150, size=len(restaurants)).clip(2000, 2800),
        index=restaurants,
    )
    # случайный шум по дням
    noise = rng.normal(1.0, 0.05, len(data))

    data["revenue"] = (
        data["guests"] * data["restaurant_id"].map(average_checks) * noise
    ).round(2)
    return data


def prepare_dataset(
    raw_dir: str | Path,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Загружает исходные данные и добавляет синтетическую выручку."""
    data = load_data(raw_dir)
    return add_synthetic_revenue(data, random_state=random_state)