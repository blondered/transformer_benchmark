import typing as tp

import numpy as np
import torch
from rectools.models.base import InternalRecoTriplet
from rectools.models.nn.transformers.similarity import SimilarityModuleBase
from rectools.models.rank import Distance, TorchRanker
from rectools.types import InternalIdsArray
from scipy import sparse

from src.models.transformers.transformer_layers.swiglu import init_feed_forward


class TwoTowerDistanceSimilarityModule(SimilarityModuleBase):
    """Two Tower similarity module."""

    dist_available: tp.List[str] = [Distance.DOT, Distance.COSINE]
    epsilon_cosine_dist: torch.Tensor = torch.tensor([1e-8])

    def __init__(
        self,
        n_factors: int,
        dropout_rate: float,
        ff_factors_multiplier: int = 4,
        bias_in_ff: bool = False,
        ff_activation: str = "swiglu",
        distance: str = "dot",
        n_ff_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.ff_blocks = []
        for _ in range(n_ff_blocks):
            self.ff_blocks.append(
                init_feed_forward(
                    n_factors,
                    ff_factors_multiplier,
                    dropout_rate,
                    ff_activation,
                    bias_in_ff,
                )
            )

        self.n_ff_blocks = n_ff_blocks

        if distance not in self.dist_available:
            raise ValueError("`dist` can only be either `dot` or `cosine`.")
        self.distance = Distance(distance)

    def _get_full_catalog_logits(
        self, session_embs: torch.Tensor, item_embs: torch.Tensor
    ) -> torch.Tensor:
        for block in self.ff_blocks:
            block.to(item_embs.device)
            item_embs = block(item_embs)

        if self.distance == Distance.COSINE:
            session_embs = self._get_embeddings_norm(session_embs)
            item_embs = self._get_embeddings_norm(item_embs)

        logits = session_embs @ item_embs.T
        return logits

    def _get_pos_neg_logits(
        self,
        session_embs: torch.Tensor,
        item_embs: torch.Tensor,
        candidate_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        # [batch_size, session_max_len, len(candidate_item_ids), n_factors]
        for block in self.ff_blocks:
            block.to(item_embs.device)
            item_embs = block(item_embs)

        if self.distance == Distance.COSINE:
            session_embs = self._get_embeddings_norm(session_embs)
            item_embs = self._get_embeddings_norm(item_embs)

        pos_neg_embs = item_embs[candidate_item_ids]
        logits = (pos_neg_embs @ session_embs.unsqueeze(-1)).squeeze(-1)
        return logits

    def _get_embeddings_norm(self, embeddings: torch.Tensor) -> torch.Tensor:
        embedding_norm = torch.norm(embeddings, p=2, dim=1).unsqueeze(dim=1)
        embeddings = embeddings / torch.max(
            embedding_norm, self.epsilon_cosine_dist.to(embeddings)
        )
        return embeddings

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
        for block in self.ff_blocks:
            block.to(item_embs.device)
            item_embs = block(item_embs)
        ranker = TorchRanker(
            distance=self.distance,
            device=item_embs.device,
            subjects_factors=user_embs[user_ids],
            objects_factors=item_embs,
        )
        user_ids_indices, all_reco_ids, all_scores = ranker.rank(
            subject_ids=np.arange(len(user_ids)),  # n_rec_users
            k=k,
            filter_pairs_csr=ui_csr_for_filter,  # [n_rec_users x n_items + n_item_extra_tokens]
            sorted_object_whitelist=sorted_item_ids_to_recommend,  # model_internal
        )
        all_user_ids = user_ids[user_ids_indices]
        return all_user_ids, all_reco_ids, all_scores
