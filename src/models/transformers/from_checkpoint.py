import typing as tp

import pandas as pd
import typing_extensions as tpe
from pydantic import BeforeValidator, PlainSerializer
from rectools.dataset import Dataset
from rectools.models.base import ErrorBehaviour, ExternalIds, ModelBase, ModelConfig
from rectools.models.nn.transformers.base import TransformerModelBase, _get_class_obj
from rectools.utils.misc import get_class_or_function_full_path

TransformerModelType = tpe.Annotated[
    tp.Type[TransformerModelBase],
    BeforeValidator(_get_class_obj),
    PlainSerializer(
        func=get_class_or_function_full_path,
        return_type=str,
        when_used="json",
    ),
]


class TransformerFromCheckpointConfig(ModelConfig):
    ckpt_path: str
    cls_type: TransformerModelType


class TransformerFromCheckpoint(ModelBase[TransformerFromCheckpointConfig]):
    config_class = TransformerFromCheckpointConfig

    def __init__(
        self, ckpt_path: str, cls_type: tp.Type[TransformerModelBase], verbose: int = 0
    ) -> None:
        super().__init__(verbose=verbose)
        self.ckpt_path = ckpt_path
        self.cls_type = cls_type
        self.model: TransformerModelBase

    def _get_config(self) -> TransformerFromCheckpointConfig:
        return TransformerFromCheckpointConfig(
            cls=self.__class__,
            ckpt_path=self.ckpt_path,
            cls_type=self.cls_type,
            verbose=self.verbose,
        )

    @classmethod
    def _from_config(cls, config: TransformerFromCheckpointConfig) -> tpe.Self:
        return cls(
            ckpt_path=config.ckpt_path,
            cls_type=config.cls_type,
            verbose=config.verbose,
        )

    def _fit(self, dataset: Dataset) -> None:
        self.model = self.cls_type.load_from_checkpoint(self.ckpt_path)

    def recommend(
        self,
        users: ExternalIds,
        dataset: Dataset,
        k: int,
        filter_viewed: bool,
        items_to_recommend: tp.Optional[ExternalIds] = None,
        add_rank_col: bool = True,
        on_unsupported_targets: ErrorBehaviour = "raise",
    ) -> pd.DataFrame:
        return self.model.recommend(
            users,
            dataset,
            k,
            filter_viewed,
            items_to_recommend,
            add_rank_col,
            on_unsupported_targets,
        )

    def recommend_to_items(  # pylint: disable=too-many-branches
        self,
        target_items: ExternalIds,
        dataset: Dataset,
        k: int,
        filter_itself: bool = True,
        items_to_recommend: tp.Optional[ExternalIds] = None,
        add_rank_col: bool = True,
        on_unsupported_targets: ErrorBehaviour = "raise",
    ) -> pd.DataFrame:
        return self.model.recommend_to_items(
            target_items,
            dataset,
            k,
            filter_itself,
            items_to_recommend,
            add_rank_col,
            on_unsupported_targets,
        )
