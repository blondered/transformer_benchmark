import typing as tp

import torch
import torch.nn as nn
from rectools.models.nn.transformers.net_blocks import (
    PreLNTransformerLayer,
    TransformerLayersBase,
)


class AlbertLayers(TransformerLayersBase):

    def __init__(
        self,
        n_blocks: int,
        n_factors: int,
        n_heads: int,
        dropout_rate: float,
        ff_factors_multiplier: int = 4,
        n_hidden_groups: int = 1,  # accept new kwarg
        n_inner_groups: int = 1,  # accept new kwarg
    ):
        super().__init__()

        self.n_blocks = n_blocks
        self.n_hidden_groups = n_hidden_groups
        self.n_inner_groups = n_inner_groups
        n_fitted_blocks = int(n_hidden_groups * n_inner_groups)
        self.transformer_blocks = nn.ModuleList(
            [
                PreLNTransformerLayer(
                    # number of encoder layer (AlBERTLayers)
                    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/albert/modeling_albert.py#L428
                    n_factors,
                    n_heads,
                    dropout_rate,
                    ff_factors_multiplier,
                )
                # https://github.com/huggingface/transformers/blob/main/src/transformers/models/albert/modeling_albert.py#L469
                for _ in range(n_fitted_blocks)
            ]
        )
        self.n_layers_per_group = n_blocks / n_hidden_groups

    def forward(
        self,
        seqs: torch.Tensor,
        timeline_mask: torch.Tensor,
        attn_mask: tp.Optional[torch.Tensor],
        key_padding_mask: tp.Optional[torch.Tensor],
    ) -> torch.Tensor:
        for block_idx in range(self.n_blocks):
            group_idx = int(block_idx / self.n_layers_per_group)
            for inner_layer_idx in range(self.n_inner_groups):
                layer_idx = group_idx * self.n_inner_groups + inner_layer_idx
                seqs = self.transformer_blocks[layer_idx](
                    seqs, attn_mask, key_padding_mask
                )
        return seqs


# TRANSFORMER_LAYERS_KWARGS = {
#     "n_hidden_groups": 2,
#     "n_inner_groups": 2,
# }
# Both BERT4Rec and SASRec can use this layers

# total number of layer which embeds are passing through: n_blocks * n_inner_groups
# total number of fitted blocks: n_hidden_groups * n_inner_groups
# hidden groups = how many different actual sequential weight blocks are there not teking into account n_blocks
