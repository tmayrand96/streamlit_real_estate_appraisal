#!/usr/bin/env python3
"""Command line helpers for the real estate appraisal project.

This module exposes a small toolkit so the Streamlit application can be
operated from a regular terminal session.  It supports three primary
actions:

* Inspect the configured regions and datasets.
* Run the trained quantile models for a CSV file or an ad-hoc example.
* Copy ("download") the generated ``.joblib`` artifacts to a local folder.

The functionality intentionally reuses the same helpers as the Streamlit
interface to guarantee parity between both entry points.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from streamlit_real_estate_appraisal import (
    QUANTILE_LEVELS,
    QUANTILE_NAME_MAP,
    REGION_CONFIG,
    create_features,
    model_path_for,
    train_quantile_models,
)

PREDICTION_COLUMN_MAP = {
    level: f"pred_{QUANTILE_NAME_MAP[level]}" for level in QUANTILE_LEVELS
}


def _coerce_value(value: str):
    """Best-effort conversion of CLI values into Python types."""
    lower = value.strip().lower()
    if lower in {"true", "yes"}:
        return 1
    if lower in {"false", "no"}:
        return 0

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _parse_key_value_pairs(pairs: Iterable[str]) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"Invalid value '{raw}'. Expected the form key=value.")
        key, value = raw.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Missing feature name in '{raw}'.")
        parsed[key] = _coerce_value(value)
    return parsed


def _load_pipeline_bundle(region_key: str, alpha: float):
    """Return a tuple containing the joblib path, pipeline and feature list."""
    model_path = model_path_for(region_key, alpha)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model for region '{region_key}' and quantile {alpha} not found at {model_path}."
        )

    blob = joblib.load(model_path)
    pipe = blob.get("pipeline")
    if pipe is None:
        model = blob.get("model")
        scaler = blob.get("scaler")
        if model is None:
            raise KeyError(
                f"Model artifact {model_path} is missing the 'pipeline' or 'model' key."
            )
        steps: List[tuple[str, object]] = []
        if scaler is not None:
            steps.append(("scaler", scaler))
        steps.append(("model", model))
        pipe = Pipeline(steps)

    features = blob.get("features")
    if features is None:
        raise KeyError(
            f"Model artifact {model_path} does not expose the feature list used during training."
        )

    return model_path, pipe, list(features)


def _prepare_inputs(region_key: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Mirror the feature engineering step used by the Streamlit UI."""
    prepared = create_features(frame, region_key=region_key, is_training=False)
    # Ensure we return a copy so later concatenations do not unexpectedly mutate
    return prepared.copy()


def run_predictions(region_key: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Run all quantile models for the provided dataframe."""
    prepared = _prepare_inputs(region_key, frame)

    predictions: Dict[str, List[float]] = {
        PREDICTION_COLUMN_MAP[level]: [] for level in QUANTILE_LEVELS
    }

    for alpha in QUANTILE_LEVELS:
        _, pipeline, features = _load_pipeline_bundle(region_key, alpha)

        missing = [col for col in features if col not in prepared.columns]
        if missing:
            raise ValueError(
                "Cannot run predictions because some required features are missing: "
                + ", ".join(missing)
            )

        preds = pipeline.predict(prepared[features])
        predictions[PREDICTION_COLUMN_MAP[alpha]].extend(preds.tolist())

    pred_frame = pd.DataFrame(predictions, index=prepared.index)
    return pred_frame


def handle_list_regions(_: argparse.Namespace) -> None:
    print("Available regions:\n------------------")
    for key, cfg in REGION_CONFIG.items():
        dataset_path = Path(cfg["data_path"])
        models_available = all(model_path_for(key, q).exists() for q in QUANTILE_LEVELS)
        print(f"{key}: {cfg['name']}")
        print(f"  Dataset : {dataset_path} ({'found' if dataset_path.exists() else 'missing'})")
        print(f"  Features: {', '.join(cfg['feature_cols'])}")
        print(f"  Models  : {'available' if models_available else 'train required'}\n")


def handle_train(args: argparse.Namespace) -> None:
    regions = args.region or list(REGION_CONFIG.keys())
    for region in regions:
        print(f"== Training models for {region} ==")
        train_quantile_models(region)


def handle_predict(args: argparse.Namespace) -> None:
    if args.from_csv and args.values:
        raise SystemExit("Please provide either --from-csv or --values, not both.")
    if not args.from_csv and not args.values:
        raise SystemExit("Provide --from-csv PATH or at least one --values key=value pair.")

    if args.from_csv:
        frame = pd.read_csv(args.from_csv)
    else:
        example = _parse_key_value_pairs(args.values)
        frame = pd.DataFrame([example])

    pred_frame = run_predictions(args.region, frame)

    combined = pd.concat([frame.reset_index(drop=True), pred_frame.reset_index(drop=True)], axis=1)

    if args.output:
        output_path = args.output.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    else:
        with pd.option_context("display.max_columns", None):
            print(combined)


def handle_download(args: argparse.Namespace) -> None:
    destination = args.destination.expanduser()
    destination.mkdir(parents=True, exist_ok=True)

    regions = args.region or list(REGION_CONFIG.keys())
    copied: List[Path] = []

    for region in regions:
        available_files: List[Path] = []
        for alpha in QUANTILE_LEVELS:
            model_path = model_path_for(region, alpha)
            if model_path.exists():
                available_files.append(model_path)
            else:
                print(f"[warning] Missing model for region {region} quantile {alpha} at {model_path}")

        if not available_files:
            continue

        if args.zip:
            archive_name = destination / f"{region.lower()}_models"
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                for src in available_files:
                    shutil.copy2(src, tmp_dir_path / src.name)
                archive_path = shutil.make_archive(str(archive_name), "zip", root_dir=tmp_dir)
            copied.append(Path(archive_path))
        else:
            for src in available_files:
                target = destination / src.name
                shutil.copy2(src, target)
                copied.append(target)

    if not copied:
        print("No model files were copied. Ensure the models have been trained.")
        return

    print("Downloaded the following files:")
    for path in copied:
        print(f"  - {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Utility commands for the real estate appraisal project.")
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list-regions", help="Show configured regions and datasets.")
    list_parser.set_defaults(func=handle_list_regions)

    train_parser = sub.add_parser("train", help="Train quantile models for one or more regions.")
    train_parser.add_argument("--region", action="append", choices=list(REGION_CONFIG.keys()), help="Region key to train. Provide multiple times for several regions.")
    train_parser.set_defaults(func=handle_train)

    predict_parser = sub.add_parser("predict", help="Run predictions using the trained joblib models.")
    predict_parser.add_argument("region", choices=list(REGION_CONFIG.keys()), help="Region key to use for the prediction.")
    predict_parser.add_argument("--from-csv", type=Path, help="CSV file containing one or more properties.")
    predict_parser.add_argument("--values", nargs="*", help="Inline feature definition in the form key=value.")
    predict_parser.add_argument("--output", type=Path, help="Optional CSV path where predictions should be saved.")
    predict_parser.set_defaults(func=handle_predict)

    download_parser = sub.add_parser("download-models", help="Copy trained .joblib artifacts to a folder.")
    download_parser.add_argument("--region", action="append", choices=list(REGION_CONFIG.keys()), help="Limit the download to specific regions. Provide multiple times for several regions.")
    download_parser.add_argument("--destination", type=Path, required=True, help="Directory where the files (or archives) will be written.")
    download_parser.add_argument("--zip", action="store_true", help="Bundle the models for each region into a ZIP archive instead of raw files.")
    download_parser.set_defaults(func=handle_download)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover - direct CLI execution
    sys.exit(main())
