import typing as tp

import torch
from rectools.models.nn.transformers.net_blocks import TransformerLayersBase
from torch import nn

from .swiglu import init_feed_forward


class LiGRLayer(nn.Module):
    """
    Transformer Layers as described in "From Features to Transformers:
    Redefining Ranking for Scalable Impact" https://arxiv.org/pdf/2502.03417

    Parameters
    ----------
    n_factors: int
        Latent embeddings size.
    n_heads: int
        Number of attention heads.
    dropout_rate: float
        Probability of a hidden unit to be zeroed.
    ff_factors_multiplier: int
        Feed-forward layers latent embedding size multiplier.
    bias_in_ff: bool
        Add bias in Linear layers of Feed Forward
    ff_activation: str
        {"swiglu", "relu", "gelu"}
    """

    def __init__(
        self,
        n_factors: int,
        n_heads: int,
        dropout_rate: float,
        ff_factors_multiplier: int = 4,
        bias_in_ff: bool = False,
        ff_activation: str = "swiglu",
    ):
        super().__init__()
        self.multi_head_attn = nn.MultiheadAttention(
            n_factors, n_heads, dropout_rate, batch_first=True
        )
        self.layer_norm_1 = nn.LayerNorm(n_factors)
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.layer_norm_2 = nn.LayerNorm(n_factors)
        self.feed_forward = init_feed_forward(
            n_factors, ff_factors_multiplier, dropout_rate, ff_activation, bias_in_ff
        )
        self.dropout_2 = nn.Dropout(dropout_rate)

        self.gating_linear_1 = nn.Linear(n_factors, n_factors)
        self.gating_linear_2 = nn.Linear(n_factors, n_factors)

    def forward(
        self,
        seqs: torch.Tensor,
        attn_mask: tp.Optional[torch.Tensor],
        key_padding_mask: tp.Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Parameters
        ----------
        seqs: torch.Tensor
            User sequences of item embeddings.
        attn_mask: torch.Tensor, optional
            Optional mask to use in forward pass of multi-head attention as `attn_mask`.
        key_padding_mask: torch.Tensor, optional
            Optional mask to use in forward pass of multi-head attention as `key_padding_mask`.


        Returns
        -------
        torch.Tensor
            User sequences passed through transformer layers.
        """
        mha_input = self.layer_norm_1(seqs)
        mha_output, _ = self.multi_head_attn(
            mha_input,
            mha_input,
            mha_input,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        gated_skip = torch.nn.functional.sigmoid(self.gating_linear_1(seqs))
        seqs = seqs + torch.mul(gated_skip, self.dropout_1(mha_output))

        ff_input = self.layer_norm_2(seqs)
        ff_output = self.feed_forward(ff_input)
        gated_skip = torch.nn.functional.sigmoid(self.gating_linear_2(seqs))
        seqs = seqs + torch.mul(gated_skip, self.dropout_2(ff_output))
        return seqs


class LiGRLayers(TransformerLayersBase):

    def __init__(
        self,
        n_blocks: int,
        n_factors: int,
        n_heads: int,
        dropout_rate: float,
        ff_factors_multiplier: int = 4,  # kwarg
        ff_activation: str = "swiglu",  # kwarg
        bias_in_ff: bool = False,  # kwarg
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.n_factors = n_factors
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.ff_factors_multiplier = ff_factors_multiplier
        self.ff_activation = ff_activation
        self.bias_in_ff = bias_in_ff
        self.transformer_blocks = nn.ModuleList(
            [self._init_transformer_block() for _ in range(self.n_blocks)]
        )

    def _init_transformer_block(self) -> nn.Module:
        return LiGRLayer(
            self.n_factors,
            self.n_heads,
            self.dropout_rate,
            self.ff_factors_multiplier,
            bias_in_ff=self.bias_in_ff,
            ff_activation=self.ff_activation,
        )

    def forward(
        self,
        seqs: torch.Tensor,
        timeline_mask: torch.Tensor,
        attn_mask: tp.Optional[torch.Tensor],
        key_padding_mask: tp.Optional[torch.Tensor],
    ) -> torch.Tensor:
        for block_idx in range(self.n_blocks):
            seqs = self.transformer_blocks[block_idx](seqs, attn_mask, key_padding_mask)
        return seqs


# TRANSFORMER_LAYERS_KWARGS = {
#     "ff_factors_multiplier": 4,
#     "bias_in_ff": True,
#     "ff_activation": "swiglu",
# }
