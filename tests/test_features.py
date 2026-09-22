import pandas as pd

from src.features import add_lag_features, add_rolling_features


def make_test_data() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=10)
    return pd.DataFrame({
        "restaurant_id": ["r1"] * 10,
        "date": dates,
        "guests": range(10, 20),
    })


def test_lag_7_uses_only_past_values() -> None:
    data = make_test_data()
    result = add_lag_features(data, lags=(7,))

    assert pd.isna(result.loc[0, "guests_lag_7"])
    assert result.loc[7, "guests_lag_7"] == 10
    assert result.loc[8, "guests_lag_7"] == 11


def test_rolling_does_not_use_current_target() -> None:
    data = make_test_data()
    result = add_rolling_features(data, windows=(3,))

    # Для даты с индексом 3 rolling должен использовать
    # guests[0], guests[1], guests[2]
    expected_mean = (10 + 11 + 12) / 3

    assert result.loc[3, "guests_roll_mean_3"] == expected_mean


def test_features_are_separated_between_restaurants() -> None:
    data = pd.DataFrame({
        "restaurant_id": ["r1"] * 4 + ["r2"] * 4,
        "date": list(pd.date_range("2025-01-01", periods=4)) * 2,
        "guests": [10, 11, 12, 13, 100, 101, 102, 103],
    })

    result = add_lag_features(data, lags=(1,))

    # Первый день каждого ресторана не должен получить
    # значение предыдущего ресторана.
    assert pd.isna(result.loc[0, "guests_lag_1"])
    assert pd.isna(result.loc[4, "guests_lag_1"])

    assert result.loc[1, "guests_lag_1"] == 10
    assert result.loc[5, "guests_lag_1"] == 100