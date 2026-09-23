"""CLI: прогноз guests на N дней вперёд по уже обученной модели.

    python predict.py --date 2017-05-01 --restaurant air_00a91d42b08b08d9
    python predict.py --date 2017-05-01 --restaurant air_00a91d42b08b08d9 --output forecast.csv

Модель не обучается здесь — обучает и сохраняет её notebooks/02_modelling.ipynb.  Этот скрипт только загружает .cbm (src.model.load_model)
и считает прогноз (src.model.forecast)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_data, load_calendar
from src.cleaning import clean_data
from src.model import load_model, forecast, ForecastError, HORIZON_DEFAULT

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "catboost_guests.cbm"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Прогноз guests на N дней вперёд")
    parser.add_argument("--date", required=True, help="первый день прогноза, YYYY-MM-DD")
    parser.add_argument("--restaurant", required=True, help="restaurant_id")
    parser.add_argument("--horizon", type=int, default=HORIZON_DEFAULT, help="число дней прогноза")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="папка с сырыми csv")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="путь к обученной модели (.cbm)")
    parser.add_argument("--output", default=None, help="сохранить прогноз в CSV вместо печати в stdout")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        start_date = pd.Timestamp(args.date)
    except (ValueError, TypeError):
        print(f"Ошибка: не могу разобрать --date={args.date!r} как дату (YYYY-MM-DD).", file=sys.stderr)
        return 1

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print(f"Ошибка: папка с исходными данными не найдена: {raw_dir}.", file=sys.stderr)
        return 1

    try:
        model = load_model(args.model_path)
        calendar = load_calendar(raw_dir)
        data = clean_data(load_data(raw_dir), calendar)
        result = forecast(
            model, data, calendar,
            restaurant_id=args.restaurant,
            start_date=start_date,
            horizon=args.horizon,
            warn=lambda msg: print(f"Внимание: {msg}", file=sys.stderr),
        )
    except ForecastError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output)
        result.to_csv(output_path, index=False)
        print(f"Прогноз сохранён в {output_path}", file=sys.stderr)
    else:
        print(result.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
