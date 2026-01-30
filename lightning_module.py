import pytorch_lightning as pl
import torch.nn as nn
import torch

class Tox21LightningModule(pl.LightningModule):
    def __init__(self, model, lr_backbone=1e-5, lr_head=5e-4):
        super().__init__()
        self.model=model
        self.loss_fn=nn.BCEWithLogitsLoss(reduction="none")
        self.lr_backbone=lr_backbone
        self.lr_head=lr_head

    def training_step(self, batch, batch_idx):
        logits=self.model(
            batch["input_ids"],
            batch["attention_mask"]
        )

        loss=self.loss_fn(logits, batch["labels"])
        loss=(loss*batch["weights"]).sum()/(batch["weights"]>0).sum()

        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits=self.model(
            batch["input_ids"],
            batch["attention_mask"]
        )

        return{
            "logits": torch.sigmoid(logits).detach(),
            "labels": batch["labels"],
            "weights": batch["weights"],
        }
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            [
                {"params": self.model.backbone.parameters(), "lr": self.lr_backbone},
                {"params": self.model.classifier.parameters(), "lr": self.lr_head},
            ]
        )
