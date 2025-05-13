import typing as tp

import torch
import torch.nn as nn
import typing_extensions as tpe
from rectools.dataset import Dataset
from rectools.dataset.dataset import DatasetSchema
from rectools.models.nn.item_net import ItemNetBase, SumOfEmbeddingsConstructor


class SumOfEmbeddingsAlbertConstructor(SumOfEmbeddingsConstructor):

    def __init__(
        self,
        n_items: int,
        n_factors: int,
        item_net_blocks: tp.Sequence[ItemNetBase],
        hidden_factors: int = 64,  # kwarg
    ) -> None:
        super().__init__(
            n_items=n_items,
            item_net_blocks=item_net_blocks,
        )
        self.item_emb_proj = nn.Linear(
            hidden_factors, n_factors
        )  # Project to actual required n_factors

    @classmethod
    def from_dataset(
        cls,
        dataset: Dataset,
        n_factors: int,
        dropout_rate: float,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]],
        hidden_factors: int,  # kwarg
    ) -> tpe.Self:
        n_items = dataset.item_id_map.size

        item_net_blocks: tp.List[ItemNetBase] = []
        for item_net in item_net_block_types:
            # Item net blocks will work in lower dimensional space
            item_net_block = item_net.from_dataset(
                dataset, hidden_factors, dropout_rate
            )
            if item_net_block is not None:
                item_net_blocks.append(item_net_block)

        return cls(n_items, n_factors, item_net_blocks, hidden_factors)

    @classmethod
    def from_dataset_schema(
        cls,
        dataset_schema: DatasetSchema,
        n_factors: int,
        dropout_rate: float,
        item_net_block_types: tp.Sequence[tp.Type[ItemNetBase]],
        hidden_factors: int,  # kwarg
    ) -> tpe.Self:
        n_items = dataset_schema.items.n_hot

        item_net_blocks: tp.List[ItemNetBase] = []
        for item_net in item_net_block_types:
            item_net_block = item_net.from_dataset_schema(
                dataset_schema, hidden_factors, dropout_rate
            )
            if item_net_block is not None:
                item_net_blocks.append(item_net_block)

        return cls(n_items, n_factors, item_net_blocks, hidden_factors)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        item_embs = super().forward(
            items
        )  # Create embeddings in lower dimensional space
        item_embs = self.item_emb_proj(
            item_embs
        )  # Project to actual required hidden space
        return item_embs


# CONSTRUCTOR_KWARGS = {
#     "hidden_factors": 64,
# }
# Both BERT4Rec and SASRec can use this constructor
