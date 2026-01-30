import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer

class SmilesDataset(Dataset):
    def __init__(
        self, 
        smiles,
        y, 
        w, 
        tokenizer,    
        max_length=128):
        self.smiles = smiles
        self.y = torch.tensor(y, dtype=torch.float32)
        self.w = torch.tensor(w, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        enc =self.tokenizer(
            self.smiles[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return{
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": self.y[idx],
            "weights": self.w[idx],
        }
    

class Tox21DataModule(pl.LightningDataModule):
    def __init__(
        self,
        model_name,
        train_data,
        valid_data,
        batch_size=16,
        num_workers=2,
        max_length=128,
    ):
        super().__init__()
        self.tokenzier = AutoTokenizer.from_pretrained(model_name)
        self.train_smiles, self.y_train, self.w_train = train_data
        self.valid_smiles, self.y_valid, self.w_valid = valid_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length

    def setup(self, stage=None):
        self.train_ds = SmilesDataset(
            self.train_smiles,
            self.y_train,
            self.w_train,
            self.tokenzier,
            max_length=self.max_length,
        )
        self.valid_ds = SmilesDataset(
            self.valid_smiles,
            self.y_valid,
            self.w_valid,
            self.tokenzier,
            max_length=self.max_length,
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
