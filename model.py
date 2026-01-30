import torch
import torch.nn as nn
from transformers import Autotokenizer, AutoModel

class DeepChemLLM(nn.Module):
    """
    DeepChem-style wrapper for HuggingFace encoder models.
    """

    def __init__(
            self,
            model_name: str,
            n_tasks: int,
            pooling: str = "cls",
            freeze_backbone: bool = True
    ):
        super().__init__()

        self.tokenizer = Autotokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, n_tasks)

        self.pooling  = pooling

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False


    def forward(self, smiles_list):
        """
        smiles_list: List[str]
        returns: logits (B, n_tasks)
        """

        device = next(self.parameters()).device


        enc = self.tokenizer(
            smiles_list,
            padding = True,
            truncation = True,
            return_tensors="pt"
        ).to(device)

        outputs = self.backbone(**enc)
        hidden = outputs.last_hidden_state

        if self.pooling =="cls":
            pooled = hidden[:,0]
        else:
            pooled = hidden.mean(dim=1)

        logits  = self.classifier(pooled)
        return logits