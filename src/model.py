from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from prophet import Prophet
from catboost import CatBoostRegressor
import pandas as pd


def build_preprocessor(
    categorical_features: list[str],
    numeric_features: list[str],
) -> ColumnTransformer:
    """Подготовка числовых и категориальных признаков."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def build_lr_model(
    preprocessor: ColumnTransformer,
    alpha: float = 1.0,
) -> Pipeline:
    """Линейная модель с L2-регуляризацией."""
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=alpha)),
    ])


def train_random_forest(preprocessor, X_train, y_train):
    """Обучение Random Forest."""
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1,
            max_depth=15,
        )),
    ])

    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train, categorical_features):
    """Обучение CatBoost."""
    model = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="MAE",
        random_seed=42,
        verbose=False,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_features,
    )

    return model


def train_prophet(
    train: pd.DataFrame,
    forecast_dates: pd.Series,
) -> pd.DataFrame:
    """Обучение Prophet и получение прогноза."""
    prophet_train = train[["date", "guests"]].rename(
        columns={"date": "ds", "guests": "y"}
    )
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.1,
    )
    model.fit(prophet_train)

    forecast = model.predict(
        pd.DataFrame({"ds": forecast_dates})
    )
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]