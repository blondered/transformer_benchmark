import logging
from pathlib import Path

import pandas as pd
from rectools import Columns

from src.datasets.common import extract_dataset, process_validation_schemes
from src.utils import console_logging

DATASET_NAME = "ml_20m"
INTERACTIONS_FILENAME = "ratings.csv"
ZIP_FILENAME = "ml-20m.zip"
URL = "http://files.grouplens.org/datasets/movielens/ml-20m.zip"
MIN_RATING = 0.0
EXTRACTED_DIRNAME = "ml-20m"
TIME_SPLIT_TEST_SIZE = "60D"


def process_raw_file(raw_data_path: Path) -> pd.DataFrame:
    """Process MovieLens 20M raw ratings file.

    Parameters
    ----------
    raw_data_path : Path
        Path to raw data directory

    Returns
    -------
    pd.DataFrame
        Processed ratings
    """
    ratings_file = raw_data_path / INTERACTIONS_FILENAME

    # Read raw file - has header by default
    ratings = pd.read_csv(ratings_file)

    # Filter by rating and rename columns
    ratings = ratings[ratings["rating"] >= MIN_RATING]
    ratings.rename(
        columns={
            "userId": Columns.User,
            "movieId": Columns.Item,
            "timestamp": Columns.Datetime,
        },
        inplace=True,
    )

    # Convert timestamp to datetime
    ratings[Columns.Datetime] = pd.to_datetime(ratings[Columns.Datetime], unit="s")
    ratings[Columns.Weight] = 1

    return ratings


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
