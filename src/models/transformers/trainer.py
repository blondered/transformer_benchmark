import typing as tp
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from rectools import Columns
from rectools.models.base import ModelBase
from scipy import sparse

ACCELERATOR = "gpu"
DEVICES = [0]

N_VAL_USERS = 2048
RECALL_K = 10

MAX_EPOCHS = 100
PATIENCE = 50
DIVERGENCE_TRESHOLD = None

SHOW_PROGRESS = True

LOGGER = CSVLogger("rectools_logs")


class BestModelLoad(Callback):

    def __init__(self, ckpt_path: str) -> None:
        self.ckpt_path = ckpt_path + ".ckpt"

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.log_dir is None:
            warnings.warn(
                "Trainer has no log dir and weights were not updated from checkpoint"
            )
            return
        log_dir: str = trainer.log_dir
        ckpt_path = Path(log_dir) / "checkpoints" / self.ckpt_path
        checkpoint = torch.load(ckpt_path, weights_only=False)
        pl_module.load_state_dict(checkpoint["state_dict"])
        self.ckpt_full_path = str(
            ckpt_path
        )  # pylint: disable = attribute-defined-outside-init


class RecallCallback(Callback):  # with filter
    name: str = "recall"

    def __init__(self, k: int, prog_bar: bool = True) -> None:
        self.k = k
        self.name += f"@{k}"
        self.prog_bar = prog_bar

        self.batch_recall_per_users: tp.List[torch.Tensor] = []

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: tp.Dict[str, torch.Tensor],
        batch: tp.Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        if "logits" not in outputs:
            session_embs = pl_module.torch_model.encode_sessions(
                batch, pl_module.item_embs
            )[:, -1, :]
            logits = pl_module.torch_model.similarity_module(
                session_embs, pl_module.item_embs
            )
        else:
            logits = outputs["logits"]

        x = batch["x"]
        users = x.shape[0]
        row_ind = np.arange(users).repeat(x.shape[1])
        col_ind = x.flatten().detach().cpu().numpy()
        mask = col_ind != 0
        data = np.ones_like(row_ind[mask])
        filter_csr = sparse.csr_matrix(
            (data, (row_ind[mask], col_ind[mask])),
            shape=(users, pl_module.torch_model.item_model.n_items),
        )
        mask = torch.from_numpy((filter_csr != 0).toarray()).to(logits.device)
        scores = torch.masked_fill(logits, mask, float("-inf"))

        _, batch_recos = scores.topk(k=self.k)

        targets = batch["y"]

        # assume all users have the same amount of TP
        liked = targets.shape[1]
        tp_mask = torch.stack(
            [
                torch.isin(batch_recos[uid], targets[uid])
                for uid in range(batch_recos.shape[0])
            ]
        )
        recall_per_users = tp_mask.sum(dim=1) / liked

        self.batch_recall_per_users.append(recall_per_users)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        recall = float(torch.concat(self.batch_recall_per_users).mean())
        self.log_dict(
            {self.name: recall}, on_step=False, on_epoch=True, prog_bar=self.prog_bar
        )

        self.batch_recall_per_users.clear()


def get_trainer() -> Trainer:
    recall_callback = RecallCallback(k=RECALL_K, prog_bar=SHOW_PROGRESS)
    max_recall_ckpt = ModelCheckpoint(
        monitor=f"recall@{RECALL_K}",
        mode="max",
        filename="best_recall",
    )
    early_stopping_recall = EarlyStopping(
        monitor=f"recall@{RECALL_K}",
        mode="max",
        patience=PATIENCE,
        divergence_threshold=DIVERGENCE_TRESHOLD,
    )

    best_model_load = BestModelLoad("best_recall")
    callbacks = [
        recall_callback,
        max_recall_ckpt,
        best_model_load,
        early_stopping_recall,
    ]
    return Trainer(
        max_epochs=MAX_EPOCHS,
        deterministic=True,
        enable_progress_bar=SHOW_PROGRESS,
        enable_model_summary=SHOW_PROGRESS,
        logger=LOGGER,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        callbacks=callbacks,
    )


def get_trainer_200_epochs() -> Trainer:
    recall_callback = RecallCallback(k=RECALL_K, prog_bar=SHOW_PROGRESS)
    max_recall_ckpt = ModelCheckpoint(
        monitor=f"recall@{RECALL_K}",
        mode="max",
        filename="best_recall",
    )
    early_stopping_recall = EarlyStopping(
        monitor=f"recall@{RECALL_K}",
        mode="max",
        patience=PATIENCE,
        divergence_threshold=DIVERGENCE_TRESHOLD,
    )

    best_model_load = BestModelLoad("best_recall")
    callbacks = [
        recall_callback,
        max_recall_ckpt,
        best_model_load,
        early_stopping_recall,
    ]
    return Trainer(
        max_epochs=200,
        deterministic=True,
        enable_progress_bar=SHOW_PROGRESS,
        enable_model_summary=SHOW_PROGRESS,
        logger=LOGGER,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        callbacks=callbacks,
    )


def get_trainer_val_loss() -> Trainer:
    min_val_loss_ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="best_val_loss",
    )
    early_stopping_val_loss = EarlyStopping(
        monitor=f"val_loss",
        mode="min",
        patience=PATIENCE,
        divergence_threshold=DIVERGENCE_TRESHOLD,
    )

    best_model_load = BestModelLoad("best_val_loss")
    callbacks = [
        min_val_loss_ckpt,
        best_model_load,
        early_stopping_val_loss,
    ]
    return Trainer(
        max_epochs=MAX_EPOCHS,
        deterministic=True,
        enable_progress_bar=SHOW_PROGRESS,
        enable_model_summary=SHOW_PROGRESS,
        logger=LOGGER,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        callbacks=callbacks,
    )


def leave_one_out_mask_for_users(
    interactions: pd.DataFrame, val_users: tp.Optional[np.ndarray] = None
) -> np.ndarray:
    rank = (
        interactions.sort_values(Columns.Datetime, ascending=False, kind="stable")
        .groupby(Columns.User, sort=False)
        .cumcount()
    )
    if val_users is not None:
        val_mask = (interactions[Columns.User].isin(val_users)) & (rank == 0)
    else:
        val_mask = rank == 0
    return val_mask.values


def get_val_mask_func_val_users_512(interactions: pd.DataFrame) -> np.ndarray:
    users = interactions[Columns.User].unique()
    val_users = users[:512]
    return leave_one_out_mask_for_users(interactions, val_users=val_users)


def get_val_mask_func_val_users_2048(interactions: pd.DataFrame) -> np.ndarray:
    users = interactions[Columns.User].unique()
    val_users = users[:2048]
    return leave_one_out_mask_for_users(interactions, val_users=val_users)


def get_val_mask_func_val_users_10000(interactions: pd.DataFrame) -> np.ndarray:
    users = interactions[Columns.User].unique()
    val_users = users[:10000]
    return leave_one_out_mask_for_users(interactions, val_users=val_users)


def get_val_mask_func_all(interactions: pd.DataFrame) -> np.ndarray:
    users = interactions[Columns.User].unique()
    return leave_one_out_mask_for_users(interactions, val_users=users)


def get_ckpt_path(model: ModelBase) -> tp.Optional[str]:
    if (
        hasattr(model, "fit_trainer")
        and isinstance(model.fit_trainer, Trainer)
        and hasattr(model.fit_trainer, "callbacks")
    ):
        callbacks = model.fit_trainer.callbacks
        for callback in callbacks:
            if isinstance(callback, BestModelLoad):
                return callback.ckpt_full_path
