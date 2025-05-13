import torch
from rectools.models.nn.transformers.net_blocks import PointWiseFeedForward
from torch import nn


class SwigluFeedForward(nn.Module):
    """
    From FuXi and LLama SwigLU https://arxiv.org/pdf/2502.03036 and LiGR https://arxiv.org/pdf/2502.03417


    Parameters
    ----------
    n_factors : int
        Latent embeddings size.
    n_factors_ff : int
        How many hidden units to use in the network.
    dropout_rate : float
        Probability of a hidden unit to be zeroed.
    bias: bool
        Add bias to Linear layers
    """

    def __init__(
        self, n_factors: int, n_factors_ff: int, dropout_rate: float, bias: bool = True
    ) -> None:
        super().__init__()
        self.ff_linear_1 = nn.Linear(n_factors, n_factors_ff, bias=bias)
        self.ff_dropout_1 = torch.nn.Dropout(dropout_rate)
        self.ff_activation = torch.nn.SiLU()
        self.ff_linear_2 = nn.Linear(n_factors_ff, n_factors, bias=bias)
        self.ff_linear_3 = nn.Linear(n_factors, n_factors_ff, bias=bias)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        seqs : torch.Tensor
            User sequences of item embeddings.

        Returns
        -------
        torch.Tensor
            User sequence that passed through all layers.
        """

        output = self.ff_activation(self.ff_linear_1(seqs)) * self.ff_linear_3(seqs)
        fin = self.ff_linear_2(self.ff_dropout_1(output))
        return fin


def init_feed_forward(
    n_factors: int,
    ff_factors_multiplier: int,
    dropout_rate: float,
    ff_activation: str,
    bias: bool = True,
) -> nn.Module:
    if ff_activation == "swiglu":
        return SwigluFeedForward(
            n_factors, n_factors * ff_factors_multiplier, dropout_rate, bias=bias
        )
    if ff_activation == "gelu":
        return PointWiseFeedForward(
            n_factors,
            n_factors * ff_factors_multiplier,
            dropout_rate,
            activation=torch.nn.GELU(),  # TODO: bias
        )
    if ff_activation == "relu":
        return PointWiseFeedForward(
            n_factors,
            n_factors * ff_factors_multiplier,
            dropout_rate,
            activation=torch.nn.ReLU(),  # TODO: bias
        )
    raise ValueError(f"Unsupported ff_activation: {ff_activation}")
