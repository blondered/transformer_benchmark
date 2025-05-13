import logging
from pathlib import Path

import pandas as pd
from rectools import Columns

from src.datasets.common import (
    apply_filtering,
    extract_dataset,
    process_interactions_ids,
    process_validation_schemes,
)
from src.utils import console_logging

DATASET_NAME = "kion_r"
INTERACTIONS_FILENAME = "interactions.csv"
ZIP_FILENAME = "kion.zip"
URL = "https://github.com/irsafilo/KION_DATASET/raw/f69775be31fa5779907cf0a92ddedb70037fb5ae/data_original.zip"
EXTRACTED_DIRNAME = "data_original"
TIME_SPLIT_TEST_SIZE = "14D"
USER_CORE = 2
ITEM_CORE = 5


def process_raw_file(raw_data_path: Path) -> pd.DataFrame:
    """Process KION raw interactions file.

    Parameters
    ----------
    raw_data_path : Path
        Path to raw data directory

    Returns
    -------
    pd.DataFrame
        Processed interactions
    """
    interactions_file = raw_data_path / INTERACTIONS_FILENAME

    # Read raw file
    interactions = pd.read_csv(interactions_file, parse_dates=["last_watch_dt"])

    # Rename columns to match rectools format
    interactions.rename(
        columns={
            "last_watch_dt": Columns.Datetime,
        },
        inplace=True,
    )

    interactions = apply_filtering(
        interactions, user_core=USER_CORE, item_core=ITEM_CORE
    )
    interactions[Columns.Weight] = 1
    dataset_path = Path("data") / DATASET_NAME
    interactions = process_interactions_ids(interactions, dataset_path)

    return interactions


if __name__ == "__main__":
    console_logging(level=logging.INFO)

    # Extract dataset (includes downloading if needed)
    extract_dataset(
        dataset_name=DATASET_NAME,
        interactions_filename=INTERACTIONS_FILENAME,
        zip_filename=ZIP_FILENAME,
        url=URL,
        extracted_dirname=EXTRACTED_DIRNAME,
    )

    # Process all validation schemes
    process_validation_schemes(
        DATASET_NAME, process_raw_file, time_split_test_size=TIME_SPLIT_TEST_SIZE
    )
