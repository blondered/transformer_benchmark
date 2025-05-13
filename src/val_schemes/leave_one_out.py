from rectools.metrics import (
    NDCG,
    AvgRecPopularity,
    CatalogCoverage,
    CoveredUsers,
    Recall,
    SufficientReco,
)

from src.evaluation.splitter import LastNSplitter

# ####  --------------  Holdout splitter  --------------  #### #

HOLDOUT_SPLITTER = LastNSplitter(
    n=1,
    n_splits=1,
    filter_cold_users=True,
    filter_cold_items=True,
    filter_already_seen=True,
)


# ####  --------------  Cross-validation splitter  --------------  #### #

CV_N_SPLITS = 1
CV_SPLITTER = LastNSplitter(
    n=1,
    n_splits=CV_N_SPLITS,
    filter_cold_users=True,
    filter_cold_items=True,
    filter_already_seen=True,
)

# ####  --------------  Metrics  --------------  #### #

K = 10
METRICS = {
    f"recall@{K}": Recall(k=K),
    f"ndcg@{K}": NDCG(k=K, divide_by_achievable=True),
    f"arp@{K}": AvgRecPopularity(k=K, normalize=True),
    f"coverage@{K}": CatalogCoverage(k=K, normalize=True),
    f"covered_users@{K}": CoveredUsers(k=K),
    f"sufficient_reco@{K}": SufficientReco(k=K),
}
