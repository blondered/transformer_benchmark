import ast
import gzip
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

DATASET_NAME = "beeradvocate_r"
INTERACTIONS_FILENAME = "beeradvocate.json"
ZIP_FILENAME = "beeradvocate.json.gz"
URL = "https://mcauleylab.ucsd.edu/public_datasets/data/beer/beeradvocate.json.gz"
TIME_SPLIT_TEST_SIZE = "60D"
USER_CORE = 2
ITEM_CORE = 5


def process_raw_file(raw_data_path: Path) -> pd.DataFrame:
    """Process Goodreads raw interactions file.

    Parameters
    ----------
    raw_data_path : Path
        Path to raw data directory

    Returns
    -------
    pd.DataFrame
        Processed interactions
    """
    interactions = []
    with gzip.open(raw_data_path / INTERACTIONS_FILENAME, "rt", encoding="utf-8") as f:
        for line in f:
            a = ast.literal_eval(line)
            if len(a) > 0:
                interactions.append(
                    {
                        "item_id": a["beer/beerId"],
                        "datetime": int(a["review/time"]),
                        "user_id": a["review/profileName"],
                        "rating": float(a["review/overall"]),
                    }
                )

    interactions = pd.DataFrame(interactions)
    interactions["datetime"] = pd.to_datetime(interactions["datetime"], unit="s")

    interactions = apply_filtering(
        interactions, user_core=USER_CORE, item_core=ITEM_CORE
    )

    interactions[Columns.Weight] = 1.0
    dataset_path = Path("data") / DATASET_NAME
    interactions = process_interactions_ids(interactions, dataset_path)

    return interactions


if __name__ == "__main__":
    console_logging(level=logging.INFO)

    # Extract dataset (includes downloading if needed)
    extract_dataset(
        dataset_name=DATASET_NAME,
        interactions_filename=INTERACTIONS_FILENAME,
        url=URL,
        zip_filename=None,
        extracted_dirname=None,  # No extraction needed
    )

    # Process all validation schemes
    process_validation_schemes(
        DATASET_NAME, process_raw_file, time_split_test_size=TIME_SPLIT_TEST_SIZE
    )
