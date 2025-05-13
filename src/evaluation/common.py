import importlib.util
import json
import logging
import os
import time
import typing as tp
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pandas as pd
import typing_extensions as tpe
from rectools import Columns
from rectools.dataset import Dataset
from rectools.metrics import calc_metrics
from rectools.model_selection import cross_validate
from rectools.models import model_from_params
from rectools.models.base import ModelConfig

from src.models.transformers.trainer import get_ckpt_path
from src.utils import get_current_commit, setup_deterministic

REPORT_PATH = "reports"
VAL_SCHEMES_PATH = "val_schemes"


class Timer:  # pylint: disable = attribute-defined-outside-init
    def __enter__(self) -> tpe.Self:
        self._start = time.perf_counter()
        self._end: tp.Optional[float] = None
        return self

    def __exit__(self, *args: tp.Any) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed(self) -> tp.Optional[float]:
        if self._end is None:
            return None
        return self._end - self._start


def get_val_scheme_arguments(val_scheme: str, attributes: List[str]) -> Dict[str, Any]:
    """Get arguments from validation scheme file.

    Parameters
    ----------
    val_scheme : str
        Validation scheme name
    attributes : List[str]
        List of attributes to get from the module

    Returns
    -------
    Dict[str, Any]
        Dictionary with requested attributes
    """
    path = os.path.join("src", VAL_SCHEMES_PATH, val_scheme + ".py")
    spec = importlib.util.spec_from_file_location(val_scheme, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {attr: getattr(module, attr) for attr in attributes}


def get_params_for_model(
    model_cls: str,
    sampled_params: Dict[str, Any],
    fixed_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Get model parameters by combining fixed and sampled parameters.

    Parameters
    ----------
    model_cls : str
        Model class name
    sampled_params : Dict[str, Any]
        Parameters sampled from search space
    fixed_params : Dict[str, Any]
        Fixed parameters

    Returns
    -------
    Dict[str, Any]
        Combined model parameters
    """
    params = {"cls": model_cls}
    params.update(fixed_params)
    params.update(sampled_params)
    return params


def get_search_options(model_config: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Get all possible parameter combinations from model config.

    Parameters
    ----------
    model_config : Dict[str, Any]
        Model configuration with search parameters

    Returns
    -------
    List[Tuple[str, Any]]
        List of parameter combinations
    """
    search_parameters = model_config["search_parameters"]
    options = []
    if search_parameters is None:
        return options

    for search_definition in search_parameters:
        name, choices = (
            search_definition["name"],
            search_definition["choices"],
        )
        options.append(tuple((name, choice) for choice in choices))

    return options


def get_report_path(val_scheme: str, dataset_name: str, report_type: str) -> str:
    """Get path for report file.

    Parameters
    ----------
    val_scheme : str
        Validation scheme name
    dataset_name : str
        Dataset name
    report_type : str
        Type of report ("cv" or "holdout")

    Returns
    -------
    str
        Path to report file
    """
    report_file_path = os.path.join(REPORT_PATH, val_scheme, dataset_name, report_type)
    os.makedirs(report_file_path, exist_ok=True)
    return report_file_path


def save_to_csv(df: pd.DataFrame, path: str) -> None:
    mode = "w"
    headers = df.columns
    if os.path.isfile(path):
        os.remove(path)
    Path(path).parent.resolve().mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, mode=mode, header=headers)


def add_missing_columns(df: pd.DataFrame, new_columns: tp.Set[str]) -> pd.DataFrame:
    for column in new_columns:
        df[column] = None
    return df


def add_to_previous_file(result_df: pd.DataFrame, metrics_path: str) -> None:
    if os.path.exists(metrics_path):
        previous_result_df = pd.read_csv(metrics_path)

        result_columns = result_df.columns
        previous_result_columns = previous_result_df.columns

        diff_columns = set(result_columns).difference(previous_result_columns)
        diff_columns_previous = set(previous_result_columns).difference(result_columns)

        result_df = add_missing_columns(result_df, diff_columns_previous)
        previous_result_df = add_missing_columns(previous_result_df, diff_columns)

        result_df = pd.concat([previous_result_df, result_df], ignore_index=True)

    save_to_csv(result_df, metrics_path)


def save_results(res: Dict[str, Any], report_file: str):
    """Save results to CSV file.

    Parameters
    ----------
    res : Dict[str, Any]
        Results to save
    report_file : str
        Path to save results
    """
    df = pd.DataFrame.from_records([res])
    add_to_previous_file(df, report_file)
    logging.info(f"Saved results to {report_file}")


def iterate_model_params(
    model_search_spec: Dict[str, Any]
) -> Iterator[Tuple[Dict[str, Any], str]]:
    """Iterate over all possible parameter combinations for a model.

    Parameters
    ----------
    model_search_spec : Dict[str, Any]
        Model configuration with search parameters

    Yields
    ------
    Dict[str, Any]
        Dictionary with model parameters for each combination
    """
    model_cls = model_search_spec["cls"]
    fixed_params = model_search_spec.get("fixed_parameters", {})
    options = get_search_options(model_search_spec)

    full_options = product(*options)
    for option in full_options:
        sampled_params = {}
        for option_detail in option:
            name, choice = option_detail
            sampled_params[name] = choice
        logging.info(sampled_params)

        current_params = get_params_for_model(model_cls, sampled_params, fixed_params)
        current_name = "_".join(
            [
                f"{key}={str(value).split('.')[-1]}"
                for key, value in sampled_params.items()
            ]
        )
        yield current_params, current_name


def validate_model_on_holdout(
    dataset_name: str,
    model_params: ModelConfig,
    val_scheme: str,
    metrics: Dict,
    k: int,
    report_file: str,
    current_name: str,
):
    """Validate model on holdout set.

    Parameters
    ----------
    dataset_name : str
        Dataset name
    model_params : ModelConfig
        Model configuration
    val_scheme : str
        Validation scheme name
    metrics : Dict
        Metrics to calculate
    k : int
        Number of recommendations
    report_file : str
        Path to save results
    """
    setup_deterministic()
    model = model_from_params(model_params)
    interactions = pd.read_csv(f"data/{dataset_name}/{val_scheme}/train.csv")
    holdout = pd.read_csv(f"data/{dataset_name}/{val_scheme}/holdout.csv")
    holdout_users = holdout[Columns.User].unique()

    dataset = Dataset.construct(interactions)
    with Timer() as timer:
        model.fit(dataset)
    fit_time = timer.elapsed
    reco = model.recommend(
        users=holdout_users,
        dataset=dataset,
        k=k,
        filter_viewed=True,
        on_unsupported_targets="warn",
    )

    metric_results = calc_metrics(
        metrics=metrics,
        reco=reco,
        interactions=holdout,
        prev_interactions=interactions,
        catalog=interactions[Columns.Item].unique(),
    )
    logging.info(metric_results)

    model_params = model.get_params(simple_types=True)
    res = {
        "comment": current_name,
        "cls": model_params["cls"],
        "dataset_name": dataset_name,
        "val_scheme": val_scheme,
        "fit_time": round(fit_time) if fit_time is not None else fit_time,
    }
    res.update(metric_results)
    res.update(
        {
            "model_params": json.dumps(model_params),
            "commit": get_current_commit(),
            "test_interactions": holdout.shape[0],
            "train_interactions": interactions.shape[0],
        }
    )
    ckpt_path = get_ckpt_path(model)
    if ckpt_path is not None:
        res["ckpt"] = ckpt_path

    save_results(res, report_file)
    logging.info(f"Completed validation for {model_params['cls']} on {dataset_name}")


def validate_model_on_cv(
    dataset_name: str,
    model_config: ModelConfig,
    val_scheme: str,
    cv_arguments: Dict,
    current_name: str,
):
    """Validate model using cross-validation.

    Parameters
    ----------
    dataset_name : str
        Dataset name
    model_config : ModelConfig
        Model configuration
    val_scheme : str
        Validation scheme name
    cv_arguments : Dict
        Cross-validation arguments

    Returns
    -------
    Dict
        Dictionary with validation results
    """
    setup_deterministic()
    model = model_from_params(model_config)
    interactions = pd.read_csv(f"data/{dataset_name}/{val_scheme}/train.csv")
    dataset = Dataset.construct(interactions)
    cv_results = cross_validate(
        dataset=dataset,
        splitter=cv_arguments["CV_SPLITTER"],
        metrics=cv_arguments["METRICS"],
        models={"model": model},
        k=cv_arguments["K"],
        filter_viewed=True,
    )
    metrics = (
        pd.DataFrame(cv_results["metrics"])
        .drop(columns=["i_split", "model"])
        .mean()
        .to_dict()
    )
    logging.info(metrics)
    model_params = model.get_params(simple_types=True)
    res = (
        {
            "comment": current_name,
            "cls": model_params["cls"],
            "dataset_name": dataset_name,
            "val_scheme": val_scheme,
        }
        | metrics
        | {
            "model_params": json.dumps(model_params),
            "commit": get_current_commit(),
            "train_interactions": interactions.shape[0],
        }
    )
    ckpt_path = get_ckpt_path(model)
    if ckpt_path is not None:
        res["ckpt"] = ckpt_path
    return res
