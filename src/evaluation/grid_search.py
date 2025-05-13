import logging
import os

from src.evaluation.common import (
    get_report_path,
    get_val_scheme_arguments,
    iterate_model_params,
    save_results,
    validate_model_on_cv,
)
from src.utils import console_logging, read_config


def validate_models_on_cv(config_file: str = "configs/grid_search/current.yaml"):
    config = read_config(config_file)
    for val_scheme in config["val_schemes"]:
        cv_arguments = get_val_scheme_arguments(
            val_scheme, ["CV_SPLITTER", "METRICS", "K"]
        )
        for dataset_name in config["datasets"]:
            report_file_path = get_report_path(val_scheme, dataset_name, "cv")
            for model_search_spec in config["models"]:
                logging.info(
                    f'Validate {model_search_spec["cls"]} on {dataset_name} using {val_scheme}'
                )

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
                    res = validate_model_on_cv(
                        dataset_name,
                        model_params,
                        val_scheme,
                        cv_arguments,
                        current_name,
                    )
                    save_results(res, report_file)


if __name__ == "__main__":
    console_logging(level=logging.INFO)
    validate_models_on_cv()
