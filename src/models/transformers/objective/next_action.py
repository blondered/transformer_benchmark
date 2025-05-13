import typing as tp

import numpy as np
import torch
from rectools.models.nn.transformers.bert4rec import BERT4RecDataPreparator
from rectools.models.nn.transformers.constants import MASKING_VALUE
from rectools.models.nn.transformers.lightning import TransformerLightningModule

# ### ---------- NextActionTransformer ---------- ### #


class NextActionDataPreparator(BERT4RecDataPreparator):

    def _collate_fn_train(
        self,
        batch: tp.List[tp.Tuple[tp.List[int], tp.List[float]]],
    ) -> tp.Dict[str, torch.Tensor]:
        """
        Truncate each session from right to keep `session_max_len` items.
        Do left padding until `session_max_len` is reached.
        Split to `x`, `y`, and `yw`.
        """
        batch_size = len(batch)
        x = np.zeros((batch_size, self.session_max_len))
        y = np.zeros((batch_size, 1))
        yw = np.zeros((batch_size, 1))
        for i, (ses, ses_weights) in enumerate(batch):
            session = ses.copy()
            session[-1] = self.extra_token_ids[
                MASKING_VALUE
            ]  # Replace last token with "MASK"
            x[i, -len(ses) :] = session
            y[i] = ses[-1]
            yw[i] = ses_weights[-1]

        batch_dict = {
            "x": torch.LongTensor(x),
            "y": torch.LongTensor(y),
            "yw": torch.FloatTensor(yw),
        }
        if self.n_negatives is not None:
            negatives = torch.randint(
                low=self.n_item_extra_tokens,
                high=self.item_id_map.size,
                size=(batch_size, 1, self.n_negatives),
            )
            batch_dict["negatives"] = negatives
        return batch_dict


class NextActionLightningModule(TransformerLightningModule):

    def training_step(
        self, batch: tp.Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        if self.loss_calculator is not None:
            y, w = batch["y"], batch["yw"]
            logits = self.get_batch_logits(batch)
            logits = logits[:, -1::]
            loss = self.loss_calculator(logits, y, w)
        else:
            loss = self._calc_custom_loss(batch, batch_idx)

        self.log(
            self.train_loss_name,
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=self.verbose > 0,
        )
        return loss


# next_action_model = BERT4RecModel(
#     data_preparator_type=NextActionDataPreparator,
#     lightning_module_type=NextActionLightningModule,
#     use_causal_attn=True  # False (both work)
# )
