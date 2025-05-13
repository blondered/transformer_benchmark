import gzip
import json
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

# From P5 repository:
# https://drive.usercontent.google.com/download?id=1uE-_wpGmIiRLxaIy8wItMspOf5xRNF2O&export=download&authuser=0
# https://github.com/jeykigung/P5/blob/main/preprocess/data_preprocess_amazon.ipynb
# https://drive.google.com/file/d/1uE-_wpGmIiRLxaIy8wItMspOf5xRNF2O/view

# https://amazon-reviews-2023.github.io/#load-user-reviews
# https://github.com/RUCAIBox/CIKM2020-S3Rec/blob/master/data/data_process.py

# Dataset variants: short name -> gzipped filename
DATASET_VARIANTS = {
    "s3_beauty_w_time": "reviews_Beauty_5.json.gz",
    "s3_toys_w_time": "reviews_Toys_and_Games_5.json.gz",
    "s3_sports_w_time": "reviews_Sports_and_Outdoors_5.json.gz",
}

URL = "https://drive.usercontent.google.com/download?id=1uE-_wpGmIiRLxaIy8wItMspOf5xRNF2O&export=download&confirm=t"
ZIP_FILENAME = "raw_data.zip"
EXTRACTED_DIRNAME = "raw_data"

MIN_RATING = 0.0
USER_CORE = 5
ITEM_CORE = 5


def process_raw_file(raw_data_path: Path, variant: str) -> pd.DataFrame:
    """Process Amazon raw ratings file for a given variant."""
    ratings_file = raw_data_path / EXTRACTED_DIRNAME / DATASET_VARIANTS[variant]

    # Read raw file and extract interactions
    interactions = []
    with gzip.open(ratings_file, "rt") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                review = json.loads(line)
                if float(review["overall"]) >= MIN_RATING:
                    interactions.append(
                        {
                            Columns.User: review["reviewerID"],
                            Columns.Item: review["asin"],
                            Columns.Datetime: int(review["unixReviewTime"]),
                            "rating": float(review["overall"]),
                        }
                    )
            except json.JSONDecodeError as e:
                logging.warning(f"Error parsing JSON: {str(e)}")
                continue
            except (KeyError, ValueError) as e:
                logging.warning(f"Error processing review: {str(e)}")
                continue

    if not interactions:
        raise ValueError(
            "No valid interactions found in the file. Please check the data format."
        )

    df = pd.DataFrame(interactions)
    df["datetime"] = pd.to_datetime(df["datetime"], unit="s")
    df = apply_filtering(df, user_core=USER_CORE, item_core=ITEM_CORE)
    df[Columns.Weight] = 1.0
    dataset_path = Path("data") / variant
    df = process_interactions_ids(df, dataset_path)
    return df


if __name__ == "__main__":
    console_logging(level=logging.INFO)

    for dataset_name, filename in DATASET_VARIANTS.items():

        # Extract dataset (includes downloading if needed)
        extract_dataset(
            dataset_name=dataset_name,
            interactions_filename=filename,
            zip_filename=ZIP_FILENAME,
            url=URL,
            extracted_dirname=EXTRACTED_DIRNAME,
        )

        # Process all validation schemes
        process_validation_schemes(
            dataset_name, lambda path: process_raw_file(path, dataset_name)
        )
