from collections import OrderedDict

from rectools.metrics import (
    NDCG,
    AvgRecPopularity,
    CatalogCoverage,
    CoveredUsers,
    DebiasConfig,
    HitRate,
    Precision,
    Recall,
    Serendipity,
    SufficientReco,
)
from rectools.model_selection import TimeRangeSplitter

FOLD_SIZE = "7D"
RANDOM_STATE = 32

# ####  --------------  Holdout splitter  --------------  #### #

HOLDOUT_SPLITTER = TimeRangeSplitter(
    test_size=FOLD_SIZE,
    n_splits=1,
    filter_cold_users=True,
    filter_cold_items=True,
    filter_already_seen=True,
)


# ####  --------------  Cross-validation splitter  --------------  #### #

CV_N_SPLITS = 1
CV_SPLITTER = TimeRangeSplitter(
    test_size=FOLD_SIZE,
    n_splits=CV_N_SPLITS,
    filter_cold_users=True,
    filter_cold_items=True,
    filter_already_seen=True,
)

# ####  --------------  Metrics  --------------  #### #

K = 10
IQR_COEFS = [1.5, 0.75, 0]
DEBIAS_CONFIGS = {
    iqr_coef: DebiasConfig(iqr_coef=iqr_coef, random_state=RANDOM_STATE)
    for iqr_coef in IQR_COEFS
}
METRICS = OrderedDict(
    [
        (f"recall@{K}", Recall(k=K)),
        (f"precision@{K}", Precision(k=K)),
        (f"hit_rate@{K}", HitRate(k=K)),
        (f"ndcg@{K}", NDCG(k=K, divide_by_achievable=True)),
        (f"arp@{K}", AvgRecPopularity(k=K, normalize=True)),
        (f"coverage@{K}", CatalogCoverage(k=K, normalize=True)),
        (f"serendipity@{K}", Serendipity(k=K)),
        (f"covered_users@{K}", CoveredUsers(k=K)),
        (f"sufficient_reco@{K}", SufficientReco(k=K)),
    ]
)

for iqr_coef in IQR_COEFS:
    METRICS[f"recall@{K}_debiased_{iqr_coef}"] = Recall(
        k=K, debias_config=DEBIAS_CONFIGS[iqr_coef]
    )
    METRICS[f"precision@{K}_debiased_{iqr_coef}"] = Precision(
        k=K, debias_config=DEBIAS_CONFIGS[iqr_coef]
    )
    METRICS[f"hit_rate@{K}_debiased_{iqr_coef}"] = HitRate(
        k=K, debias_config=DEBIAS_CONFIGS[iqr_coef]
    )
    METRICS[f"ndcg@{K}_debiased_{iqr_coef}"] = NDCG(
        k=K, divide_by_achievable=True, debias_config=DEBIAS_CONFIGS[iqr_coef]
    )
