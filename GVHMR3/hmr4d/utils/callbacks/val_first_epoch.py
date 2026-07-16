import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from hmr4d.utils.pylogger import Log
from hmr4d.configs import MainStore, builds


class ValFirstEpoch(Callback):
    """Force a validation pass at the end of epoch 0, then restore the user's
    original `check_val_every_n_epoch` so subsequent val runs follow the
    config-defined cadence (e.g. every 10 epochs).

    Mechanism:
      - on_fit_start: stash trainer.check_val_every_n_epoch and set it to 1
        so Lightning's `(current_epoch + 1) % N == 0` check fires after epoch 0.
      - on_validation_epoch_end (first non-sanity call): restore the stashed
        value so subsequent epochs follow the user's intended cadence.
    """

    def __init__(self):
        super().__init__()
        self._original_n = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._original_n = trainer.check_val_every_n_epoch
        trainer.check_val_every_n_epoch = 1
        Log.info(
            f"[ValFirstEpoch] forcing val after epoch 0 "
            f"(original check_val_every_n_epoch={self._original_n})"
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Skip sanity-check val (also defensive in case num_sanity_val_steps>0).
        if getattr(trainer, "sanity_checking", False):
            return
        # Restore exactly once, on the first real val pass (end of epoch 0).
        if self._original_n is not None:
            trainer.check_val_every_n_epoch = self._original_n
            Log.info(
                f"[ValFirstEpoch] epoch-0 val done; "
                f"restored check_val_every_n_epoch={self._original_n}"
            )
            self._original_n = None


group_name = "callbacks/val_first_epoch"
MainStore.store(name="base", node=builds(ValFirstEpoch), group=group_name)
