import gzip
import logging
import os
import shlex
import shutil
import subprocess
import typing as tp
from pathlib import Path

import pandas as pd
import requests
from rectools import Columns
from rectools.dataset.interactions import IdMap, Interactions
from rectools.model_selection import TimeRangeSplitter
from rectools.utils.misc import import_object
from tqdm import tqdm

VAL_SCHEMES_PATH = Path("src") / "val_schemes"
RAW_DATA_DIR = Path("data/raw")
INTERACTIONS_SAVE_DIR = Path("data")


def shell(cmd):
    """Run a shell command.

    Parameters
    ----------
    cmd : str
        Command to run
    """
    logging.info("running shell command: \n {}".format(cmd))
    subprocess.check_call(shlex.split(cmd))


def get_dir():
    """Get the library directory path.

    Returns
    -------
    str
        Path to the library directory
    """
    utils_dirname = os.path.dirname(os.path.abspath(__file__))
    lib_dirname = os.path.abspath(os.path.join(utils_dirname, ".."))
    return lib_dirname


def mkdir_p(dir_path):
    """Create directory if it doesn't exist.

    Parameters
    ----------
    dir_path : str
        Directory path to create
    """
    shell("mkdir -p {}".format(dir_path))


def mkdir_p_local(relative_dir_path):
    """Create folder inside of library if it doesn't exist.

    Parameters
    ----------
    relative_dir_path : str
        Relative path to create

    Returns
    -------
    str
        Absolute path to created directory
    """
    local_dir = get_dir()
    abspath = os.path.join(local_dir, relative_dir_path)
    mkdir_p(abspath)
    return abspath


def download_file(url, filename, data_dir):
    """Download a file if it doesn't exist.

    Parameters
    ----------
    url : str
        URL to download from
    filename : str
        Name to save the file as
    data_dir : str
        Directory to save the file in

    Returns
    -------
    str
        Path to the downloaded file
    """
    mkdir_p_local(data_dir)
    full_filename = os.path.join(get_dir(), data_dir, filename)
    if not os.path.isfile(full_filename):
        if (
            url.startswith("https://disk.yandex.ru/")
            or url.startswith("https://www.kaggle.com")
            or url.startswith("https://drive.google.com")
        ):
            dataset_name = Path(data_dir).name
            raise RuntimeError(
                f"Dataset {dataset_name} doesn't have a public link. "
                f"Please download it manually from {url} "
                f"and place it in data/raw/{dataset_name}/ directory."
            )

        logging.info(f"downloading {filename} file")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 KB

        with open(full_filename, "wb") as out_file, tqdm(
            desc=filename,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                size = out_file.write(data)
                pbar.update(size)

        logging.info(f"{filename} dataset downloaded")
    else:
        logging.info(f"{filename} file already exists, skipping")
    return full_filename


def get_val_scheme(file: str, val_schemes_path: Path):
    """Get validation scheme from file name.

    Parameters
    ----------
    file : str
        Validation scheme file name
    val_schemes_path : Path
        Path to validation schemes directory

    Returns
    -------
    tuple
        (val_scheme, splitter) pair
    """
    val_scheme = Path(file).stem
    module_path = f"src.{val_schemes_path.name}.{val_scheme}.HOLDOUT_SPLITTER"
    splitter = import_object(module_path)
    return val_scheme, splitter


def update_loo_correction(dataset_path: Path, val_scheme: str) -> None:
    """Update leave-one-out correction value in statistics after processing leave_one_out validation scheme.

    Parameters
    ----------
    dataset_path : Path
        Path to dataset directory
    val_scheme : str
        Validation scheme name (should be 'leave_one_out')
    """
    # Check for validation scheme folder
    scheme_path = dataset_path / val_scheme
    if not scheme_path.exists():
        return

    # Check for holdout.csv
    holdout_path = scheme_path / "holdout.csv"
    if not holdout_path.exists():
        return

    # Check for statistics.csv
    stats_path = dataset_path / "statistics.csv"
    if not stats_path.exists():
        logging.warning(f"No statistics.csv found in {dataset_path}")
        return

    try:
        # Count holdout rows
        holdout_df = pd.read_csv(holdout_path)
        n_holdout_rows = len(holdout_df)

        # Read existing statistics
        stats_df = pd.read_csv(stats_path)

        # Calculate loo_correction
        n_users = int(stats_df["n_users"].iloc[0])
        loo_correction = n_holdout_rows / n_users

        # Update statistics with loo_correction
        stats_df[f"loo_correction_{val_scheme}"] = loo_correction

        # Save updated statistics
        stats_df.to_csv(stats_path, index=False)

        logging.info(
            f"Updated {dataset_path.name} statistics: loo_correction={loo_correction:.4f} for {val_scheme}"
        )
    except Exception as e:
        logging.error(f"Error updating statistics for {dataset_path.name}: {str(e)}")


def calc_validation_stats(dataset_path: Path, val_scheme: str) -> None:
    """Calculate and update validation scheme statistics.

    Parameters
    ----------
    dataset_path : Path
        Path to dataset directory
    val_scheme : str
        Validation scheme name
    """
    # Check for validation scheme folder
    scheme_path = dataset_path / val_scheme
    if not scheme_path.exists():
        return

    # Check for required files
    train_path = scheme_path / "train.csv"
    holdout_path = scheme_path / "holdout.csv"
    stats_path = dataset_path / "statistics.csv"

    if not train_path.exists() or not holdout_path.exists():
        return

    if not stats_path.exists():
        logging.warning(f"No statistics.csv found in {dataset_path}")
        return

    try:
        # Read train and holdout files
        train_df = pd.read_csv(train_path)
        holdout_df = pd.read_csv(holdout_path)

        # Calculate statistics
        n_train = len(train_df)
        n_holdout = len(holdout_df)
        holdout_ratio = n_holdout / (n_train + n_holdout)

        # Read existing statistics
        stats_df = pd.read_csv(stats_path)

        # Update statistics
        stats_df[f"{val_scheme}_n_train_interactions"] = n_train
        stats_df[f"{val_scheme}_n_holdout_interactions"] = n_holdout
        stats_df[f"{val_scheme}_holdout_ratio"] = holdout_ratio

        # Save updated statistics
        stats_df.to_csv(stats_path, index=False)

        logging.info(
            f"Updated {dataset_path.name} validation statistics for {val_scheme}: "
            f"train={n_train}, holdout={n_holdout}, ratio={holdout_ratio:.4f}"
        )
    except Exception as e:
        logging.error(
            f"Error updating validation statistics for {dataset_path.name}: {str(e)}"
        )


def process_validation_schemes(
    dataset_name: str,
    process_raw_file,
    select_val_schemes: tp.List[str] = None,
    time_split_test_size: tp.Optional[str] = None,
):
    """Process all validation schemes for the dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    process_raw_file : callable
        Function that knows how to read and process the raw file
    select_val_schemes : tp.List[str], optional
        List of validation schemes to process. If None, process all schemes.
        Default is None.
    time_split_test_size : str, optional
        Test size for time_split validation scheme. If provided, overrides the default test_size.
        Default is None.
    """
    raw_data_path = RAW_DATA_DIR / dataset_name
    interactions_save_path = INTERACTIONS_SAVE_DIR / dataset_name

    # Read interactions once for all schemes
    interactions_df = process_raw_file(raw_data_path)

    # Calculate and save statistics once
    calculate_statistics(interactions_df, interactions_save_path)

    for file in VAL_SCHEMES_PATH.iterdir():
        if file.name != "__init__.py" and file.suffix == ".py":
            if select_val_schemes and file.name not in select_val_schemes:
                continue
            val_scheme, splitter = get_val_scheme(file.name, VAL_SCHEMES_PATH)

            # Override test_size for time_split if specified
            if val_scheme == "time_split" and time_split_test_size is not None:
                new_splitter = TimeRangeSplitter(
                    test_size=time_split_test_size,
                    n_splits=splitter.n_splits,
                    filter_cold_users=splitter.filter_cold_users,
                    filter_cold_items=splitter.filter_cold_items,
                    filter_already_seen=splitter.filter_already_seen,
                )
                splitter = new_splitter

            process_and_save_interactions(
                interactions=interactions_df,
                save_path=interactions_save_path,
                val_scheme=val_scheme,
                splitter=splitter,
            )

            # Update validation statistics for all schemes
            calc_validation_stats(interactions_save_path, val_scheme)

            # Update loo_correction only for leave_one_out scheme
            if val_scheme == "leave_one_out":
                update_loo_correction(interactions_save_path, val_scheme)


def extract_archive(archive_path: Path, output_path: Path, archive_type: str = None):
    """Extract an archive file.

    Parameters
    ----------
    archive_path : Path
        Path to the archive file
    output_path : Path
        Path to extract to
    archive_type : str, optional
        Type of archive ('zip' or 'gz'). If None, inferred from file extension.
        Default is None.
    """
    if archive_type is None:
        archive_type = archive_path.suffix.lstrip(".")

    if archive_type == "zip":
        shell(f"unzip -o {archive_path} -d {output_path}")
    elif archive_type == "gz":
        # For .gz files, use Python's gzip module
        output_file = output_path / archive_path.stem  # Remove .gz extension
        with gzip.open(archive_path, "rb") as f_in:
            with open(output_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        raise ValueError(f"Unsupported archive type: {archive_type}")


def extract_dataset(
    dataset_name: str,
    interactions_filename: str,
    url: str,
    zip_filename: tp.Optional[str] = None,
    extracted_dirname: str = None,
):
    """Extract dataset from archive file.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    interactions_filename : str
        Name of the interactions file in the archive
    url : str
        URL to download the dataset from
    zip_filename : str | None, optional
        Name of the archive file. If None, no extraction will be performed.
        Default is None.
    extracted_dirname : str, optional
        Name of the directory created after extraction.
        If None, uses dataset_name. Default is None.
    """
    raw_data_path = RAW_DATA_DIR / dataset_name

    # Construct file paths
    interactions_file = raw_data_path / interactions_filename
    if zip_filename is not None and extracted_dirname is not None:
        interactions_file = raw_data_path / extracted_dirname / interactions_filename
        
    archive_file = raw_data_path / zip_filename if zip_filename else None

    if interactions_file.is_file():
        logging.info("dataset is already extracted")
        return

    # Download dataset if needed
    download_file(url, zip_filename or interactions_filename, f"../{raw_data_path}")

    # Extract dataset if zip_filename is provided
    if zip_filename is not None:
        extract_archive(archive_file, raw_data_path)

    # Handle directory structure if needed (for zip files with directories)
    if extracted_dirname:
        dataset_dir = raw_data_path / extracted_dirname
        if dataset_dir.exists():
            for filename in dataset_dir.iterdir():
                shell(f"mv {filename} {raw_data_path}")
            shell(f"rm -rf {dataset_dir}")


def calculate_statistics(interactions_df: pd.DataFrame, save_path: Path):
    """Calculate dataset statistics.

    Parameters
    ----------
    interactions_df : pd.DataFrame
        DataFrame with user-item interactions
    save_path : Path
        Path to save statistics

    Returns
    -------
    dict
        Dictionary with statistics
    """
    n_users = interactions_df[Columns.User].nunique()
    n_items = interactions_df[Columns.Item].nunique()
    n_interactions = len(interactions_df)
    avg_length = interactions_df.groupby(Columns.User).size().mean()
    sparsity = 1 - (n_interactions / (n_users * n_items))

    # Temporal statistics
    min_date = interactions_df[Columns.Datetime].min()
    max_date = interactions_df[Columns.Datetime].max()
    # dates_period = (max_date - min_date).days

    stats = {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "avg_length": avg_length,
        "sparsity": sparsity,
        "min_date": min_date,  # Keep as raw timestamp
        # 'min_date': min_date.strftime('%Y-%m-%d'),  # Format as date string
        "max_date": max_date,  # Keep as raw timestamp
        # 'max_date': max_date.strftime('%Y-%m-%d'),  # Format as date string
        # 'dates_period_days': dates_period
    }

    # Save statistics
    save_path.mkdir(parents=True, exist_ok=True)
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(save_path / "statistics.csv", index=False)

    logging.info(f"Dataset statistics: {stats}")
    return stats


def process_and_save_interactions(
    interactions: pd.DataFrame, save_path: Path, val_scheme: str, splitter
):
    """Process interactions and save train/holdout interactions.

    Parameters
    ----------
    interactions : pd.DataFrame
        DataFrame with interactions
    save_path : Path
        Path to save processed files
    val_scheme : str
        Validation scheme name
    splitter : Splitter
        Splitter instance
    """
    if (save_path / val_scheme / "train.csv").is_file() and (
        save_path / val_scheme / "holdout.csv"
    ).is_file():
        logging.info(f"Interactions already splitted and saved for scheme {val_scheme}")
        return

    logging.info(f"Preparing interactions for val scheme {val_scheme}")

    # Split and save
    split_iterator = splitter.split(Interactions(interactions))
    train_ids, test_ids, _ = next(iter(split_iterator))
    train = interactions.iloc[train_ids]
    holdout = interactions.iloc[test_ids]

    logging.info(
        f"Train date range: {train[Columns.Datetime].min()} to {train[Columns.Datetime].max()}"
    )
    logging.info(
        f"Holdout date range: {holdout[Columns.Datetime].min()} to {holdout[Columns.Datetime].max()}"
    )

    scheme_dir = save_path / val_scheme
    scheme_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(scheme_dir / "train.csv", index=False)
    holdout.to_csv(scheme_dir / "holdout.csv", index=False)


def process_interactions_ids(
    interactions_df: pd.DataFrame, dataset_path: Path, keep_extra_cols: bool = True
) -> Interactions:
    """Create Interactions object with ID maps and save them to files.

    Parameters
    ----------
    interactions_df : pd.DataFrame
        DataFrame with user-item interactions
    dataset_path : Path
        Path to dataset directory where ID maps will be saved
    keep_extra_cols : bool, optional
        Whether to keep extra columns in the Interactions object.
        Default is True.

    Returns
    -------
    Interactions
        Created Interactions object with ID maps
    """
    # Create ID maps
    user_id_map = IdMap.from_values(interactions_df[Columns.User].values)
    item_id_map = IdMap.from_values(interactions_df[Columns.Item].values)

    # Create Interactions object
    interactions = Interactions.from_raw(
        interactions_df, user_id_map, item_id_map, keep_extra_cols=keep_extra_cols
    )

    # Ensure dataset directory exists
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Create id_maps directory
    id_maps_dir = dataset_path / "id_maps"
    id_maps_dir.mkdir(parents=True, exist_ok=True)

    # Save user ID map
    user_id_map_path = id_maps_dir / "user_id_map.csv"
    pd.DataFrame({"original_id": user_id_map.to_external.values}).to_csv(
        user_id_map_path, index=True
    )

    # Save item ID map
    item_id_map_path = id_maps_dir / "item_id_map.csv"
    pd.DataFrame({"original_id": item_id_map.to_external.values}).to_csv(
        item_id_map_path, index=True
    )

    logging.info(f"Saved ID maps to {id_maps_dir}")

    return interactions.df


def apply_filtering(
    interactions: pd.DataFrame, user_core: int, item_core: int
) -> pd.DataFrame:
    """Apply core filtering to interactions data.

    Iteratively removes users and items with fewer than user_core/item_core interactions
    until convergence is reached.

    Parameters
    ----------
    interactions : pd.DataFrame
        Input interactions DataFrame
    user_core : int
        Minimum number of interactions required for a user to be kept
    item_core : int
        Minimum number of interactions required for an item to be kept

    Returns
    -------
    pd.DataFrame
        Filtered interactions DataFrame
    """
    logging.info(
        f"Starting {user_core}-core filtering for users and {item_core}-core filtering for items"
    )

    while True:
        # Count interactions per user and item
        user_counts = interactions[Columns.User].value_counts()
        item_counts = interactions[Columns.Item].value_counts()

        # Find users and items to remove
        users_to_remove = user_counts[user_counts < user_core].index
        items_to_remove = item_counts[item_counts < item_core].index

        if len(users_to_remove) == 0 and len(items_to_remove) == 0:
            break

        # Log progress
        n_users_before = interactions[Columns.User].nunique()
        n_items_before = interactions[Columns.Item].nunique()
        n_interactions_before = len(interactions)

        # Remove interactions with filtered users and items
        interactions = interactions[
            ~interactions[Columns.User].isin(users_to_remove)
            & ~interactions[Columns.Item].isin(items_to_remove)
        ]

        # Log changes
        n_users_after = interactions[Columns.User].nunique()
        n_items_after = interactions[Columns.Item].nunique()
        n_interactions_after = len(interactions)

        logging.info(
            f"Filtered iteration: "
            f"users {n_users_before} -> {n_users_after} "
            f"items {n_items_before} -> {n_items_after} "
            f"interactions {n_interactions_before} -> {n_interactions_after}"
        )

    logging.info(f"Core filtering completed")
    return interactions
