import logging
from pathlib import Path

import pandas as pd
from rectools import Columns

from src.datasets.common import extract_dataset, process_validation_schemes
from src.utils import console_logging

DATASET_VARIANTS = {
    "beauty": "Beauty",
    "sports": "Sports_and_Outdoors",
    "toys": "Toys_and_Games",
}

URL_TEMPLATE = (
    "https://raw.githubusercontent.com/RUCAIBox/CIKM2020-S3Rec/master/data/{}.txt"
)


def process_raw_file(raw_data_path: Path, dataset_variant: str) -> pd.DataFrame:
    """Process S3 Amazon/LastFM/Yelp dataset raw file.

    Parameters
    ----------
    raw_data_path : Path
        Path to raw data directory
    dataset_variant : str
        Dataset variant name (e.g. 'beauty', 'lastfm', etc.)

    Returns
    -------
    pd.DataFrame
        Processed interactions
    """
    if dataset_variant not in DATASET_VARIANTS:
        raise ValueError(f"Unknown dataset variant: {dataset_variant}")

    interactions_file = raw_data_path / f"{dataset_variant}.txt"

    # Read raw file
    actions = []
    with open(interactions_file) as input:
        for line in input:
            user, items = line.strip().split(" ", 1)
            items = items.split(" ")
            current_timestamp = 0
            for item in items:
                current_timestamp += 1
                actions.append(
                    {"user_id": user, "item_id": item, "datetime": current_timestamp}
                )

    interactions = pd.DataFrame(actions)
    interactions[Columns.Weight] = 1
    interactions[Columns.Item] = interactions[Columns.Item].astype(int)
    interactions[Columns.User] = interactions[Columns.User].astype(int)
    return interactions


if __name__ == "__main__":
    console_logging(level=logging.INFO)

    # Process all variants
    for variant in DATASET_VARIANTS:
        dataset_name = f"s3_{variant}"
        interactions_filename = f"{variant}.txt"

        # Extract dataset (includes downloading if needed)
        extract_dataset(
            dataset_name=dataset_name,
            interactions_filename=interactions_filename,
            zip_filename=None,
            url=URL_TEMPLATE.format(DATASET_VARIANTS[variant]),
            extracted_dirname=None,
        )

        # Process all validation schemes
        process_validation_schemes(
            dataset_name,
            lambda path: process_raw_file(path, variant),
            select_val_schemes=["leave_one_out.py"],
        )
