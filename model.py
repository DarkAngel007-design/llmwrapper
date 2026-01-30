import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model

class DeepChemLLM(nn.Module):
    """
    DeepChem-style wrapper for HuggingFace encoder models.
    Supports:
    - frozen backbone
    - full finetuning
    - QLoRA (4-bit + LoRA)
    """

    def __init__(
            self,
            model_name: str,
            n_tasks: int,
            pooling: str = "cls",
            freeze_backbone: bool = False,
            qlora: bool = False,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pooling = pooling

        if qlora:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                load_in_4bit = True,
                torch_dtype=torch.float16,
                device_map= {"": 0},
                )
            
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=["query", "key", "value"],
                task_type="FEATURE_EXTRACTION",
            )

            self.backbone = get_peft_model(self.backbone, lora_config)

        else:
            self.backbone  = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, n_tasks)


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



