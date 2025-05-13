import typing as tp
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from rectools import Columns
from rectools.models.nn.transformers.base import ValMaskCallable
from rectools.models.nn.transformers.data_preparator import SequenceDataset
from rectools.models.nn.transformers.negative_sampler import (
    TransformerNegativeSamplerBase,
)
from rectools.models.nn.transformers.sasrec import SASRecDataPreparator
from torch.utils.data import DataLoader


class DenseAllActionSequenceDataset(SequenceDataset):

    def __init__(
        self,
        sessions: tp.List[tp.List[int]],
        weights: tp.List[tp.List[float]],
        possible_targets_idx: tp.List[tp.List[tp.List[int]]],
    ):
        super().__init__(
            sessions=sessions,
            weights=weights,
        )
        # === Changed code ===
        self.possible_targets_idx = possible_targets_idx
        # === Changed code ===

    def __getitem__(
        self, index: int
    ) -> tp.Tuple[tp.List[int], tp.List[float], tp.List[tp.List[tp.List[int]]]]:
        session, weights = super().__getitem__(index)
        # === Changed code ===
        possible_targets_idx = self.possible_targets_idx[index]
        # === Changed code ===
        return session, weights, possible_targets_idx

    # === Changed code ===
    @classmethod
    def from_interactions(
        cls,
        interactions: pd.DataFrame,
        targets_window_days: str,
        sort_users: bool = False,
    ) -> "DenseAllActionSequenceDataset":
        sessions = (
            interactions.sort_values(Columns.Datetime, kind="stable")
            .groupby(Columns.User, sort=sort_users)[
                [Columns.Item, Columns.Weight, Columns.Datetime]
            ]
            .agg(list)
        )
        sessions, weights, datetimes = (
            sessions[Columns.Item].to_list(),
            sessions[Columns.Weight].to_list(),
            sessions[Columns.Datetime].to_list(),
        )

        # Selecting possible target indices
        # Last item of the session doesn't have target, as it can be only target not `x`
        possible_targets_idx = []
        for ses_idx, ses in enumerate(sessions):
            ses_datetimes = datetimes[ses_idx]
            idx_ses_datetimes_mapping = list(enumerate(ses_datetimes))
            ses_targets_idx = []
            for idx, min_y_dt in enumerate(ses_datetimes):
                max_y_dt = min_y_dt + timedelta(days=targets_window_days)
                targets_idx = list(
                    map(
                        lambda x: x[0],
                        filter(
                            lambda x: x[1] > min_y_dt and x[1] <= max_y_dt,
                            idx_ses_datetimes_mapping,
                        ),
                    )
                )
                if len(targets_idx) < 1 and idx + 1 < len(ses):
                    targets_idx = [idx + 1]
                ses_targets_idx.append(targets_idx)
            possible_targets_idx.append(ses_targets_idx)

        return cls(
            sessions=sessions,
            weights=weights,
            possible_targets_idx=possible_targets_idx,
        )

    # === Changed code ===


class DenseAllActionDataPreparator(SASRecDataPreparator):
    """Data preparator for SASRecModel."""

    def __init__(
        self,
        session_max_len: int,
        batch_size: int,
        dataloader_num_workers: int,
        train_min_user_interactions: int,
        targets_window_days: int = 14,  # kwargs
        shuffle_train: bool = True,
        get_val_mask_func: tp.Optional[ValMaskCallable] = None,
        n_negatives: tp.Optional[int] = None,
        negative_sampler: tp.Optional[TransformerNegativeSamplerBase] = None,
    ) -> None:
        super().__init__(
            session_max_len=session_max_len,
            n_negatives=n_negatives,
            batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            train_min_user_interactions=train_min_user_interactions,
            negative_sampler=negative_sampler,
            shuffle_train=shuffle_train,
            get_val_mask_func=get_val_mask_func,
        )
        # === Changed code ===
        self.targets_window_days = targets_window_days
        # === Changed code ===

    def get_dataloader_train(self) -> DataLoader:
        # === Changed code ===
        sequence_dataset = DenseAllActionSequenceDataset.from_interactions(
            self.train_dataset.interactions.df, self.targets_window_days
        )
        # === Changed code ===
        train_dataloader = DataLoader(
            sequence_dataset,
            collate_fn=self._collate_fn_train,
            batch_size=self.batch_size,
            num_workers=self.dataloader_num_workers,
            shuffle=self.shuffle_train,
        )
        return train_dataloader

    # === Changed code ===
    def _collate_fn_train(
        self,
        batch: List[Tuple[List[int], List[float]]],
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        x = np.zeros((batch_size, self.session_max_len))
        y = np.zeros((batch_size, self.session_max_len))
        yw = np.zeros((batch_size, self.session_max_len))
        for i, (ses, ses_weights, ses_possible_target_indices) in enumerate(batch):
            x[i, -len(ses) + 1 :] = ses[
                :-1
            ]  # ses: [session_len] -> x[i]: [session_max_len]

            # Sampling targets from possible indices.
            # `do not aim to predict all positive actions, and instead only aim to predict one positive action`
            start_y_index = -len(ses) + 1
            for position, possible_target_indices in enumerate(
                ses_possible_target_indices
            ):
                if len(possible_target_indices) == 0:
                    continue
                target_index = np.random.choice(possible_target_indices, size=1)[0]
                y[i, start_y_index + position] = ses[target_index]
                yw[i, start_y_index + position] = ses_weights[target_index]

        batch_dict = {
            "x": torch.LongTensor(x),
            "y": torch.LongTensor(y),
            "yw": torch.FloatTensor(yw),
        }
        if self.negative_sampler is not None:
            batch_dict["negatives"] = self.negative_sampler.get_negatives(
                batch_dict,
                lowest_id=self.n_item_extra_tokens,
                highest_id=self.item_id_map.size,
            )
        return batch_dict

    # === Changed code ===


# DATA_PREPARATOR_KWARGS = {
#     "targets_window_days": 7,
# }

# dense_all_action_model = SASRecModel(
#     data_preparator_type=DenseAllActionDataPreparator,
#     data_preparator_kwargs=DATA_PREPARATOR_KWARGS,
#     use_causal_attn=True,  # as in paper PinnerFormer but suddenly False also works
# )
