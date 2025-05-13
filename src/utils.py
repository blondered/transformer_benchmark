import logging
import os
import subprocess  # nosec
import typing as tp
import warnings

import threadpoolctl
import yaml


def read_config(config_file: str) -> tp.Dict:
    with open(config_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config


def console_logging(level=logging.DEBUG) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    )


def get_current_commit() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    )  # nosec


def setup_deterministic(random_seed: int = 32) -> None:
    """Setup deterministic behavior for PyTorch and other libraries.

    Parameters
    ----------
    random_seed : int, optional
        Random seed to use, by default 32
    """
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    threadpoolctl.threadpool_limits(1, "blas")

    try:
        import torch
        from lightning_fabric import seed_everything

        torch.use_deterministic_algorithms(True)
        seed_everything(random_seed, workers=True)

        # Enable deterministic behaviour with CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except ImportError as e:
        warnings.warn(f"Failed to import PyTorch: {e}")
