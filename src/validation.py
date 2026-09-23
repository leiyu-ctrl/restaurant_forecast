import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def temporal_split(
    data: pd.DataFrame,
    validation_weeks: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Разделить данные по времени без перемешивания."""
    split_date = data["date"].max() - pd.Timedelta(weeks=validation_weeks)

    train = data[data["date"] < split_date].copy()
    valid = data[data["date"] >= split_date].copy()

    return train, valid


def evaluate_predictions(
    y_true: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Посчитать MAE, RMSE и MAPE."""
    mae = mean_absolute_error(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))

    mask = y_true != 0
    mape = np.mean(
        np.abs(
            (y_true[mask] - predictions[mask])
            / y_true[mask]
        )
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
    }