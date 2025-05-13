import typing as tp
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from rectools import Columns
from rectools.dataset import Dataset, Interactions
from rectools.dataset.identifiers import IdMap
from rectools.models.nn.transformers.base import ValMaskCallable
from rectools.models.nn.transformers.bert4rec import BERT4RecDataPreparator
from rectools.models.nn.transformers.constants import MASKING_VALUE
from rectools.models.nn.transformers.lightning import TransformerLightningModule
from rectools.models.nn.transformers.negative_sampler import (
    TransformerNegativeSamplerBase,
)
from rectools.models.nn.transformers.torch_backbone import TransformerTorchBackbone


class AllActionDataPreparator(BERT4RecDataPreparator):
    """Data preparator for SASRecModel."""

    def __init__(
        self,
        session_max_len: int,
        n_negatives: tp.Optional[int],
        batch_size: int,
        dataloader_num_workers: int,
        train_min_user_interactions: int,
        negative_sampler: tp.Optional[TransformerNegativeSamplerBase] = None,
        mask_prob: float = 0.15,
        shuffle_train: bool = True,
        get_val_mask_func: tp.Optional[ValMaskCallable] = None,
        last_k_days: int = 7,  # kwargs
        max_k_actions: int = 32,  # kwargs
        random_state: int = 17,  # kwargs
    ) -> None:
        super().__init__(
            session_max_len=session_max_len,
            n_negatives=n_negatives,
            negative_sampler=negative_sampler,
            batch_size=batch_size,
            dataloader_num_workers=dataloader_num_workers,
            train_min_user_interactions=train_min_user_interactions,
            shuffle_train=shuffle_train,
            get_val_mask_func=get_val_mask_func,
            mask_prob=mask_prob,
        )
        # === Changed code ===
        self.last_k_days = last_k_days
        self.max_k_actions = max_k_actions
        self.random_state = random_state
        # === Changed code ===

    # === Changed code ===
    def get_target_mask_func(
        self,
        interactions: pd.DataFrame,
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        min_train_date = interactions[Columns.Datetime].max() - timedelta(
            days=self.last_k_days
        )
        split_mask = interactions[Columns.Datetime] > min_train_date
        masked_interactions = interactions[split_mask]
        target_indixes = (
            masked_interactions.groupby(Columns.User, sort=False)
            .sample(self.max_k_actions, replace=True, random_state=self.random_state)
            .drop_duplicates(keep="first")
            .index
        )
        target_mask = pd.Series(False, index=range(len(interactions)))
        target_mask.loc[target_indixes] = True

        target_mask = target_mask.values
        train_mask = ~split_mask.values
        return target_mask, train_mask

    # === Changed code ===

    def process_dataset_train(self, dataset: Dataset) -> None:
        """Process train dataset and save data."""
        raw_interactions = dataset.get_raw_interactions()

        # Exclude val interaction targets from train if needed
        interactions = raw_interactions
        if self.get_val_mask_func is not None:
            val_mask = self.get_val_mask_func(raw_interactions)
            interactions = raw_interactions[~val_mask]
            # So that indixes don't mismatch, when are sampled targets (line 65-66)
            interactions.reset_index(drop=True, inplace=True)

        # Filter train interactions
        # === Changed code ===
        if self.get_target_mask_func is not None:
            target_mask, train_mask = self.get_target_mask_func(interactions)

            train_interactions = interactions[train_mask]
            train_interactions = self._filter_train_interactions(train_interactions)
            train_targets = interactions[target_mask]

            # Filtering of users who are not in one of the samples
            users_intersections = np.intersect1d(
                train_interactions[Columns.User].unique(),
                train_targets[Columns.User].unique(),
            )
            train_interactions = train_interactions[
                train_interactions[Columns.User].isin(users_intersections)
            ]
            train_targets = train_targets[
                train_targets[Columns.User].isin(users_intersections)
            ]

            train_interactions[Columns.Weight] = 0
            interactions = pd.concat([train_interactions, train_targets], axis=0)
        else:
            interactions = self._filter_train_interactions(interactions)
        # === Changed code ===

        # Prepare id maps
        user_id_map = IdMap.from_values(interactions[Columns.User].values)
        item_id_map = IdMap.from_values(self.item_extra_tokens)
        item_id_map = item_id_map.add_ids(interactions[Columns.Item])

        # Prepare item features
        item_features = None
        if dataset.item_features is not None:
            item_features = self._process_features_for_id_map(
                dataset.item_features,
                dataset.item_id_map,
                item_id_map,
                self.n_item_extra_tokens,
            )

        # Prepare train dataset
        # User features are dropped for now because model doesn't support them
        final_interactions = Interactions.from_raw(
            interactions, user_id_map, item_id_map, keep_extra_cols=True
        )
        self.train_dataset = Dataset(
            user_id_map, item_id_map, final_interactions, item_features=item_features
        )
        self.item_id_map = self.train_dataset.item_id_map
        self._init_extra_token_ids()

        # Define val interactions
        if self.get_val_mask_func is not None:
            val_targets = raw_interactions[val_mask]
            val_targets = val_targets[
                (val_targets[Columns.User].isin(user_id_map.external_ids))
                & (val_targets[Columns.Item].isin(item_id_map.external_ids))
            ]
            val_interactions = interactions[
                interactions[Columns.User].isin(val_targets[Columns.User].unique())
            ].copy()
            val_interactions[Columns.Weight] = 0
            val_interactions = pd.concat([val_interactions, val_targets], axis=0)
            self.val_interactions = Interactions.from_raw(
                val_interactions, user_id_map, item_id_map
            ).df

    # === Changed code ===
    @staticmethod
    def get_max_target_length(
        batch: List[Tuple[List[int], List[float]]],
    ) -> int:
        max_target_length = 0
        for _, ses_weights in batch:
            target_length = len([weight for weight in ses_weights if weight != 0])
            max_target_length = max(max_target_length, target_length)
        return max_target_length

    # === Changed code ===

    def _collate_fn_train(
        self,
        batch: List[Tuple[List[int], List[float]]],
    ) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)

        # === Changed code ===
        max_target_length = self.get_max_target_length(batch)

        x = np.zeros((batch_size, self.session_max_len))
        y = np.zeros((batch_size, max_target_length))
        yw = np.zeros((batch_size, max_target_length))
        for i, (ses, ses_weights) in enumerate(batch):
            train_session = [
                ses[idx] for idx, weight in enumerate(ses_weights) if weight == 0
            ] + [self.extra_token_ids[MASKING_VALUE]]
            target_indices = [
                idx for idx, weight in enumerate(ses_weights) if weight != 0
            ]

            # ses: [session_len] -> x[i]: [session_max_len]
            x[i, -len(train_session) :] = train_session[-self.session_max_len :]
            # weights are sorted by values from 0 to 1, so it is sufficient to get only the initial targeting ids
            y[i, -len(target_indices) :] = ses[
                target_indices[0] :
            ]  # y[i]: [max_target_length]
            yw[i, -len(target_indices) :] = ses_weights[
                target_indices[0] :
            ]  # yw[i]: [max_target_length]

        batch_dict = {
            "x": torch.LongTensor(x),
            "y": torch.LongTensor(y),
            "yw": torch.FloatTensor(yw),
            "max_target_length": torch.tensor(max_target_length),
        }
        if self.negative_sampler is not None:
            batch_dict["negatives"] = self.negative_sampler.get_negatives(
                batch_dict,
                lowest_id=self.n_item_extra_tokens,
                highest_id=self.item_id_map.size,
                session_len_limit=max_target_length,
            )
        return batch_dict
        # === Changed code ===


class AllActionLightningModule(TransformerLightningModule):

    def get_batch_logits(
        self, batch: tp.Dict[str, torch.Tensor], training_step: bool = False
    ) -> torch.Tensor:
        """Get bacth logits."""
        if self._requires_negatives:
            y, negatives = batch["y"], batch["negatives"]
            # if negatives.shape[1] > y.shape[1]:
            #     diff = negatives.shape[1] - y.shape[1]
            #     negatives = negatives[:, :-diff, :]
            pos_neg = torch.cat([y.unsqueeze(-1), negatives], dim=-1)
            if training_step:
                max_target_length = batch["max_target_length"]
                logits = self.torch_model(
                    batch=batch,
                    candidate_item_ids=pos_neg,
                    max_target_length=max_target_length,
                )
            else:
                logits = self.torch_model(batch=batch, candidate_item_ids=pos_neg)
        else:
            logits = self.torch_model(batch=batch)
        return logits

    def training_step(
        self, batch: tp.Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        if self.loss_calculator is not None:
            y, w, max_target_length = (
                batch["y"],
                batch["yw"],
                batch["max_target_length"],
            )
            logits = self.get_batch_logits(batch, training_step=True)
            if not self._requires_negatives:
                logits = logits[:, -1::].repeat(1, 1, max_target_length, 1).squeeze()
            loss = self.loss_calculator(logits, y, w)
        else:
            loss = self._calc_custom_loss(batch, batch_idx)

        self.log(
            self.train_loss_name,
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=self.verbose > 0,
        )

        return loss


class AllActionTransformerTorchBackbone(TransformerTorchBackbone):

    def forward(
        self,
        batch: tp.Dict[str, torch.Tensor],  # batch["x"]: [batch_size, session_max_len]
        candidate_item_ids: tp.Optional[torch.Tensor] = None,
        max_target_length: tp.Optional[int] = None,
    ) -> torch.Tensor:
        item_embs = (
            self.item_model.get_all_embeddings()
        )  # [n_items + n_item_extra_tokens, n_factors]
        session_embs = self.encode_sessions(
            batch, item_embs
        )  # [batch_size, session_max_len, n_factors]
        # === Condition for train stage with negative sampling ===
        # Also we can repeat every session `max_target_length` time, but it isn't optimal
        # This solution reduce compution costs.
        if max_target_length is not None:
            session_embs = (
                session_embs[:, -1:, :].repeat(1, 1, max_target_length, 1).squeeze()
            )
        logits = self.similarity_module(session_embs, item_embs, candidate_item_ids)
        return logits


# DATA_PREPARATOR_KWARGS = {
#     "last_k_days": 7,  # kwargs
#     "max_k_actions": 32,  # kwargs
#     "random_state": 17,  # kwargs
# }

# all_action_model = BERT4RecModel(
#     data_preparator_type=AllActionDataPreparator,
#     lightning_module_type=AllActionLightningModule,
#     data_preparator_kwargs=DATA_PREPARATOR_KWARGS,
#     backbone_type=AllActionTransformerTorchBackbone,
#     use_causal_attn=True  # False (both work)
# )
