import argparse
import logging
import os
from pathlib import Path

from src.evaluation.common import (
    get_report_path,
    get_val_scheme_arguments,
    iterate_model_params,
    validate_model_on_holdout,
)
from src.utils import console_logging, read_config, setup_deterministic


def validate_models_on_holdout(
    config_file: str = "configs/holdout/current_params.yaml",
):
    """Validate models on holdout set for all parameter combinations.

    Parameters
    ----------
    config_file : str
        Path to config file with model parameters
    """
    config = read_config(config_file)

    for val_scheme in config["val_schemes"]:
        # Get metrics and k from validation scheme
        scheme_args = get_val_scheme_arguments(val_scheme, ["METRICS", "K"])
        metrics, k = scheme_args["METRICS"], scheme_args["K"]

        for dataset_name in config["datasets"]:
            for model_search_spec in config["models"]:
                model_cls = model_search_spec["cls"]
                logging.info(
                    f"Validate {model_cls} on {dataset_name} using {val_scheme}"
                )

                # Get report path
                report_file_path = get_report_path(val_scheme, dataset_name, "holdout")
                report_file = os.path.join(
                    report_file_path, model_search_spec["report_file_name"] + ".csv"
                )

                for model_params, search_name in iterate_model_params(
                    model_search_spec
                ):
                    if len(search_name) > 0:
                        current_name = model_search_spec["comment"] + "_" + search_name
                    else:
                        current_name = model_search_spec["comment"]
                    validate_model_on_holdout(
                        dataset_name=dataset_name,
                        model_params=model_params,
                        val_scheme=val_scheme,
                        metrics=metrics,
                        k=k,
                        report_file=report_file,
                        current_name=current_name,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        help="Path to config file",
        default="configs/holdout/current_params.yaml",
    )
    args = parser.parse_args()

    console_logging(level=logging.INFO)
    setup_deterministic()
    validate_models_on_holdout(config_file=args.config_file)
