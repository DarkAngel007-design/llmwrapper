import pytorch_lightning as pl
import torch.nn as nn
import torch
import numpy as np

from llmwrapper.eval import evaluate_multitask


class Tox21LightningModule(pl.LightningModule):
    def __init__(self, model, lr_backbone=1e-5, lr_head=5e-4):
        super().__init__()
        self.model=model
        self.loss_fn=nn.BCEWithLogitsLoss(reduction="none")
        self.lr_backbone=lr_backbone
        self.lr_head=lr_head

        self.validation_outputs = []

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
            "probs": torch.sigmoid(logits).detach().cpu().numpy(),
            "labels": batch["labels"].cpu().numpy(),
            "weights": batch["weights"].cpu().numpy(),
        }
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        self.validation_outputs.append(outputs)

    def on_validation_epoch_end(self):
        if len(self.validation_outputs) == 0:
            print("WARNING: No validation outputs collected!")
            return
        
        y_pred = np.concatenate([o["probs"] for o in self.validation_outputs], axis=0)
        y_true = np.concatenate([o["labels"] for o in self.validation_outputs], axis=0)
        w = np.concatenate([o["weights"] for o in self.validation_outputs], axis=0)

        metrics = evaluate_multitask(y_true, y_pred, w)

        if metrics["roc_auc"] is not None:
            self.log("val_roc_auc", metrics["roc_auc"], prog_bar=True, on_epoch=True)
            self.log("val_pr_auc", metrics["pr_auc"], prog_bar=True, on_epoch=True)
            self.log("val_n_tasks", metrics["n_valid_tasks"], on_epoch=True)

        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            [
                {"params": self.model.backbone.parameters(), "lr": self.lr_backbone},
                {"params": self.model.classifier.parameters(), "lr": self.lr_head},
            ]
        )
