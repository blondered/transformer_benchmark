from torch import nn

from .albert import AlbertLayers
from .ligr import LiGRLayer


class AlLiGRLayers(AlbertLayers):

    def __init__(
        self,
        n_blocks: int,
        n_factors: int,
        n_heads: int,
        dropout_rate: float,
        ff_factors_multiplier: int = 4,  # kwarg
        n_hidden_groups: int = 1,  # kwarg
        n_inner_groups: int = 1,  # kwarg
        ff_activation: str = "swiglu",  # kwarg
        bias_in_ff: bool = False,  # kwarg
    ):
        self.bias_in_ff = bias_in_ff
        self.ff_activation = ff_activation

        super().__init__(
            n_blocks=n_blocks,
            n_factors=n_factors,
            n_heads=n_heads,
            dropout_rate=dropout_rate,
            ff_factors_multiplier=ff_factors_multiplier,
            n_hidden_groups=n_hidden_groups,
            n_inner_groups=n_inner_groups,
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


# TRANSFORMER_LAYERS_KWARGS = {
#     "n_hidden_groups": 1,
#     "n_inner_groups": 1,
#     "ff_factors_multiplier": 4,
#     "bias_in_ff": True,
#     "ff_activation": "swiglu",
# }
