import logging
from pathlib import Path

import pandas as pd
from rectools import Columns

from src.datasets.common import extract_dataset, process_validation_schemes
from src.utils import console_logging

URL_TEMPLATE = "https://raw.githubusercontent.com/asash/BERT4rec_py3_tf2/master/BERT4rec/data/{filename}"
DATASETS = {
    # "beauty": "beauty.txt",
    "ml_1m": "ml-1m.txt",
    # "steam": "steam.txt"
}


def process_raw_file(raw_data_path: Path, dataset_name: str) -> pd.DataFrame:
    """Process BERT4Rec dataset raw interactions file.

    Parameters
    ----------
    raw_data_path : Path
        Path to raw data directory
    dataset_name : str
        Name of the dataset

    Returns
    -------
    pd.DataFrame
        Processed interactions
    """
    interactions_file = raw_data_path / DATASETS[dataset_name]

    # Read raw file
    actions = []
    prev_user = None
    current_timestamp = 0

    with open(interactions_file) as input:
        for line in input:
            user, item = [int(id) for id in line.strip().split()]
            if user != prev_user:
                current_timestamp = 0
            prev_user = user
            current_timestamp += 1
            actions.append(
                {"user_id": user, "item_id": item, "datetime": current_timestamp}
            )

    interactions = pd.DataFrame(actions)
    interactions[Columns.Weight] = 1

    return interactions


if __name__ == "__main__":
    console_logging(level=logging.INFO)

    # Process all BERT4Rec datasets
    for dataset_name, filename in DATASETS.items():
        dataset_name_repro = f"{dataset_name}_repro"

        # Extract dataset (includes downloading if needed)
        extract_dataset(
            dataset_name=dataset_name_repro,
            interactions_filename=filename,
            zip_filename=None,
            url=URL_TEMPLATE.format(filename=filename),
            extracted_dirname=None,
        )

        # Process all validation schemes
        process_validation_schemes(
            dataset_name_repro,
            lambda path: process_raw_file(path, dataset_name),
            select_val_schemes=["leave_one_out.py"],
        )
