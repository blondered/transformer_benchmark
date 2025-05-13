import json
import logging
import os

import pandas as pd

from src.evaluation.common import (
    REPORT_PATH,
    get_val_scheme_arguments,
    validate_model_on_holdout,
)
from src.utils import console_logging, read_config, setup_deterministic


def validate_models_from_cv_reports(
    config_file: str = "configs/holdout/current_report.yaml",
):
    """Validate models on holdout set using best parameters from CV reports.

    Parameters
    ----------
    config_file : str
        Path to config file with CV report configurations
    """
    # Read the config file
    config = read_config(config_file)

    # Iterate over all report configurations
    for report_config in config["cv_reports"]:
        # Extract values from config
        goal_metric = report_config["goal_metric"]
        dataset_name = report_config["dataset_name"]
        cv_report_path = report_config["cv_report_path"]
        val_scheme = report_config["val_scheme"]
        model_cls = report_config["cls"]
        holdout_report_file_name = report_config["holdout_report_file_name"]

        # Get metrics from validation scheme
        scheme_args = get_val_scheme_arguments(val_scheme, ["METRICS", "K"])
        metrics, k = scheme_args["METRICS"], scheme_args["K"]

        # Construct report paths
        cv_report_file = os.path.join(
            REPORT_PATH, val_scheme, dataset_name, "cv", cv_report_path
        )
        report_file_path = os.path.join(
            REPORT_PATH, val_scheme, dataset_name, "holdout"
        )
        os.makedirs(report_file_path, exist_ok=True)
        report_file = os.path.join(report_file_path, f"{holdout_report_file_name}.csv")

        logging.info(f"Validating {model_cls} on {dataset_name} using {val_scheme}")

        report = pd.read_csv(cv_report_file)
        report = report[
            (report["cls"] == model_cls) & (report["dataset_name"] == dataset_name)
        ]
        report.sort_values(by=goal_metric, ascending=False, inplace=True)
        model_params = json.loads(report["model_params"][0])

        validate_model_on_holdout(
            dataset_name=dataset_name,
            model_params=model_params,
            val_scheme=val_scheme,
            metrics=metrics,
            k=k,
            report_file=report_file,
            current_name=f"best_{model_cls}_by_{goal_metric}",
        )


if __name__ == "__main__":
    console_logging(level=logging.INFO)
    setup_deterministic()
    validate_models_from_cv_reports()
