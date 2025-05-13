import typing as tp
from collections.abc import Hashable

import torch
from rectools import Columns, ExternalIds
from rectools.dataset.dataset import DatasetSchemaDict
from rectools.models.nn.transformers.data_preparator import (
    TransformerDataPreparatorBase,
)
from rectools.models.nn.transformers.lightning import TransformerLightningModule
from rectools.models.nn.transformers.negative_sampler import (
    CatalogUniformSampler,
    TransformerNegativeSamplerBase,
)
from rectools.models.nn.transformers.torch_backbone import TransformerBackboneBase


class InBatchSampler(TransformerNegativeSamplerBase):
    """Class to sample negatives uniformly from all catalog items."""

    def __init__(
        self,
        n_negatives: int,
        **kwargs: tp.Any,
    ) -> None:
        super().__init__(n_negatives)

    def get_negatives(
        self,
        batch_dict: tp.Dict,
        lowest_id: int,
        highest_id: int,
        session_len_limit: tp.Optional[int] = None,
    ) -> torch.Tensor:
        """Return sampled negatives."""
        session_len = (
            session_len_limit
            if session_len_limit is not None
            else batch_dict["x"].shape[1]
        )
        batch_items = torch.unique(batch_dict["x"].flatten())[lowest_id:]
        neg_inds = torch.randint(
            0,
            len(batch_items),
            size=(batch_dict["x"].shape[0], session_len, self.n_negatives),
        )
        negatives = batch_items[neg_inds]
        # replace negatives equal to target
        mask = negatives == batch_dict["y"].unsqueeze(-1)
        replacement_inds = torch.randint(
            0,
            len(batch_items),
            size=(batch_dict["x"].shape[0], session_len, self.n_negatives),
        )
        negatives[mask] = batch_items[replacement_inds][mask]
        return negatives


MIXED_KWARGS = {"ratio": 0.5}


class MixedSampler(TransformerNegativeSamplerBase):
    """Class to sample negatives uniformly from all catalog items."""

    def __init__(
        self,
        n_negatives: int,
        ratio: float = 0.5,
        **kwargs: tp.Any,
    ) -> None:
        super().__init__(n_negatives)
        self.in_batch_sampler = InBatchSampler(int(n_negatives * ratio))
        self.uniform_sampler = CatalogUniformSampler(
            n_negatives - int(n_negatives * ratio)
        )
        self.ratio = ratio

    def get_negatives(
        self,
        batch_dict: tp.Dict,
        lowest_id: int,
        highest_id: int,
        session_len_limit: tp.Optional[int] = None,
    ) -> torch.Tensor:
        """Return sampled negatives."""
        in_batch_negatives = self.in_batch_sampler.get_negatives(
            batch_dict, lowest_id, highest_id, session_len_limit
        )
        uniform_negatives = self.uniform_sampler.get_negatives(
            batch_dict, lowest_id, highest_id, session_len_limit
        )
        return torch.cat([in_batch_negatives, uniform_negatives], dim=-1)


class LogQLightningModule(TransformerLightningModule):

    def __init__(
        self,
        torch_model: TransformerBackboneBase,
        model_config: tp.Dict[str, tp.Any],
        dataset_schema: DatasetSchemaDict,
        item_external_ids: ExternalIds,
        item_extra_tokens: tp.Sequence[Hashable],
        data_preparator: TransformerDataPreparatorBase,
        lr: float,
        gbce_t: float,
        loss: str,
        verbose: int = 0,
        train_loss_name: str = "train_loss",
        val_loss_name: str = "val_loss",
        adam_betas: tp.Tuple[float, float] = (0.9, 0.98),
        **kwargs: tp.Any,
    ):
        super().__init__(
            torch_model,
            model_config,
            dataset_schema,
            item_external_ids,
            item_extra_tokens,
            data_preparator,
            lr,
            gbce_t,
            loss,
            verbose,
            train_loss_name,
            val_loss_name,
            adam_betas,
        )
        self.sampling_probs = self._calc_sampling_probs()

    def _calc_sampling_probs(self) -> None:
        items = torch.tensor(
            self.data_preparator.train_dataset.interactions.df[Columns.Item]
        )
        _, counts = torch.unique(items, sorted=True, return_counts=True)
        return torch.cat(
            (
                torch.full((self.data_preparator.n_item_extra_tokens,), 1),
                counts / torch.tensor(len(items)),
            ),
            dim=-1,
        )

    def get_batch_logits(self, batch: tp.Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get bacth logits."""
        if self._requires_negatives:
            y, negatives = batch["y"], batch["negatives"]
            pos_neg = torch.cat([y.unsqueeze(-1), negatives], dim=-1)
            logits = self.torch_model(batch=batch, candidate_item_ids=pos_neg)
            # Log-Q correction
            # leave padding unchanged, small logits should be fine for padding
            # mask = (y != 0).unsqueeze(-1).expand(-1, -1, pos_neg.shape[-1]).float()
            # use fitem frequency in training dataset as sampling probability
            sampling_probs = self.sampling_probs.to(self.device)[pos_neg]
            logits -= torch.log(sampling_probs)  # * mask
        else:
            logits = self.torch_model(batch=batch)
        return logits
