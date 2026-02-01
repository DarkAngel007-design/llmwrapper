import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy  as np

class RegressionLightningModule(pl.LightningModule):
    """
    LightningModule for regression tasks (ESOL, FreeSolv).
    """

    def __init__(self, model,lr_backbone=1e-5, lr_head=5e-4):
        super().__init__()
        self.model=model
        self.loss_fn=nn.MSELoss()
        self.lr_backbone=lr_backbone
        self.lr_head=lr_head
        self.validation_outputs=[]

    def training_step(self, batch, batch_idx):
        preds = self.model(
            batch["input_ids"],
            batch["attention_mask"]
        )
        loss = self.loss_fn(preds, batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds = self.model(
            batch["input_ids"],
            batch["attention_mask"]
        )
        return {
            "preds": preds.detach().cpu().numpy(),
            "labels": batch["labels"].cpu().numpy(),
        }

    def on_validation_batch_end(self, outputs, *args):
        self.validation_outputs.append(outputs)

    def on_validation_epoch_end(self):
        y_pred = np.concatenate([o["preds"] for o in self.validation_outputs])
        y_true = np.concatenate([o["labels"] for o in self.validation_outputs])

        rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
        mae = np.abs(y_pred - y_true).mean()

        self.log("val_rmse", rmse, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)

        self.validation_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [
                {"params": self.model.backbone.parameters(), "lr": self.lr_backbone},
                {"params": self.model.classifier.parameters(), "lr": self.lr_head},
            ]
        )