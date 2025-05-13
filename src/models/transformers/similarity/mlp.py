import logging
import typing as tp

import numpy as np
import torch
from rectools.models.base import InternalRecoTriplet
from rectools.models.nn.transformers.similarity import SimilarityModuleBase
from rectools.types import InternalIdsArray
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DeepNetwork(nn.Module):

    def __init__(
        self,
        in_features: int,
        deep_layers_dim: list,
        dropout_rate: float,
        ln: bool = False,
    ):
        super().__init__()

        deep_layers_dim = [in_features] + deep_layers_dim
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(deep_layers_dim[i], deep_layers_dim[i + 1])
                for i in range(len(deep_layers_dim) - 1)
            ]
        )
        bn = nn.LayerNorm if ln else nn.BatchNorm1d
        self.batch_norm = nn.ModuleList(
            [bn(deep_layers_dim[i + 1]) for i in range(len(deep_layers_dim) - 1)]
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        embs: torch.Tensor,
    ) -> torch.Tensor:
        for layer_idx in range(len(self.linear_layers)):
            embs = self.activation(
                self.batch_norm[layer_idx](self.linear_layers[layer_idx](embs))
            )
            embs = self.dropout(embs)
        return embs


class MLPSimilarityModule(SimilarityModuleBase):
    """MLP similarity module."""

    def __init__(
        self, n_factors, dropout_rate, deep_layers_dim: tp.Optional = None
    ) -> None:
        super().__init__()
        if deep_layers_dim is None:
            deep_layers_dim = []
        in_features = n_factors * 2
        self.mlp = DeepNetwork(
            in_features, deep_layers_dim + [1], dropout_rate, ln=True
        )

    def _get_full_catalog_logits(
        self, session_embs: torch.Tensor, item_embs: torch.Tensor
    ) -> torch.Tensor:
        # session_embs: [batch x session_max_len x n_factors]
        # item_embs: [n_items x n_factors]

        # No session dim -> make it a dummy
        if len(session_embs.shape) == 2:
            session_embs = session_embs.unsqueeze(1)

        batch_size, session_max_len, _ = session_embs.shape

        n_items = item_embs.shape[0]

        # 1. Expand session embeddings to match desired dimensions
        # [batch x session_max_len x 1 x n_factors] -> [batch x session_max_len x n_items x n_factors]
        expanded_session_embs = session_embs.unsqueeze(2).expand(-1, -1, n_items, -1)

        # 2. Expand item embeddings to match desired dimensions
        # [1 x 1 x n_items x n_factors] -> [batch x session_max_len x n_items x n_factors]
        expanded_item_embs = (
            item_embs.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, session_max_len, -1, -1)
        )

        # 3. Concatenate along the last dimension to get [batch x session_max_len x n_items x n_factors*2]
        mlp_input = torch.cat([expanded_session_embs, expanded_item_embs], dim=-1)

        logits = self.mlp(mlp_input).squeeze()  # [batch x session_max_len x n_items]

        return logits

    def _get_pos_neg_logits(
        self,
        session_embs: torch.Tensor,
        item_embs: torch.Tensor,
        candidate_item_ids: torch.Tensor,
    ) -> torch.Tensor:

        if (
            candidate_item_ids.shape[1] == 1
        ):  # val dataloader with negatives only for last session position
            session_embs = session_embs[:, -1, :].unsqueeze(1)

        # [batch_size, session_max_len, len(candidate_item_ids), n_factors]
        pos_neg_embs = item_embs[candidate_item_ids]

        n_items = pos_neg_embs.shape[2]

        # 1. Expand session embeddings to match desired dimensions
        # [batch x session_max_len x 1 x n_factors] -> [batch x session_max_len x n_items x n_factors]
        expanded_session_embs = session_embs.unsqueeze(2).expand(-1, -1, n_items, -1)

        # 3. Concatenate along the last dimension to get [batch x session_max_len x n_items x n_factors*2]
        mlp_input = torch.cat([expanded_session_embs, pos_neg_embs], dim=-1)

        logits = self.mlp(mlp_input).squeeze()  # [batch x session_max_len x n_items]

        # Return session dim
        if candidate_item_ids.shape[1] == 1:
            logits = logits.unsqueeze(1)

        return logits

    def forward(
        self,
        session_embs: torch.Tensor,
        item_embs: torch.Tensor,
        candidate_item_ids: tp.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass to get logits."""

        if candidate_item_ids is None:
            return self._get_full_catalog_logits(session_embs, item_embs)
        return self._get_pos_neg_logits(session_embs, item_embs, candidate_item_ids)

    def _recommend_u2i(
        self,
        user_embs: torch.Tensor,
        item_embs: torch.Tensor,
        user_ids: InternalIdsArray,
        k: int,
        sorted_item_ids_to_recommend: InternalIdsArray,
        ui_csr_for_filter: tp.Optional[sparse.csr_matrix],
    ) -> InternalRecoTriplet:
        """Recommend to users."""

        if sorted_item_ids_to_recommend is not None:
            item_embs = item_embs[sorted_item_ids_to_recommend]
        n_items = item_embs.shape[0]

        assert user_embs.shape[0] == len(user_ids)

        user_embs_dataset = TensorDataset(torch.arange(user_embs.shape[0]), user_embs)
        dataloader = DataLoader(user_embs_dataset, batch_size=128, shuffle=False)
        mask_values = float("-inf")
        all_top_scores_list = []
        all_top_inds_list = []
        all_target_inds_list = []
        with torch.no_grad():
            for (
                cur_user_ids,
                cur_user_embs,
            ) in dataloader:
                # [batch x 1 x n_factors] -> [batch x n_items x n_factors]
                expanded_session_embs = cur_user_embs.unsqueeze(1).expand(
                    -1, n_items, -1
                )

                # [1 x n_items x n_factors] -> [batch x n_items x n_factors]
                expanded_item_embs = item_embs.unsqueeze(0).expand(
                    expanded_session_embs.shape[0], -1, -1
                )
                mlp_input = torch.cat(
                    [expanded_session_embs, expanded_item_embs], dim=-1
                )
                scores = self.mlp(mlp_input).squeeze()  # [batch x n_items]

                if ui_csr_for_filter is not None:
                    mask = (
                        torch.from_numpy(
                            ui_csr_for_filter[cur_user_ids].toarray()[
                                :, ui_csr_for_filter
                            ]
                        ).to(scores.device)
                        != 0
                    )
                    scores = torch.masked_fill(scores, mask, mask_values)

                top_scores, top_inds = torch.topk(
                    scores,
                    k=min(k, scores.shape[1]),
                    dim=1,
                    sorted=True,
                    largest=True,
                )
                all_top_scores_list.append(top_scores.cpu().numpy())
                all_top_inds_list.append(top_inds.cpu().numpy())
                all_target_inds_list.append(cur_user_ids.cpu().numpy())

        all_top_scores = np.concatenate(all_top_scores_list, axis=0)
        all_top_inds = np.concatenate(all_top_inds_list, axis=0)
        all_target_inds = np.concatenate(all_target_inds_list, axis=0)

        # flatten and convert inds back to input ids
        all_scores = all_top_scores.flatten()
        all_target_ids = all_target_inds.repeat(all_top_inds.shape[1])
        all_reco_ids = sorted_item_ids_to_recommend[all_top_inds].flatten()

        # filter masked items if they appeared at top
        if ui_csr_for_filter is not None:
            mask = all_scores > mask_values
            all_scores = all_scores[mask]
            all_target_ids = all_target_ids[mask]
            all_reco_ids = all_reco_ids[mask]

        return (all_target_ids, all_reco_ids, all_scores)
