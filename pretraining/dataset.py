from torch.utils.data import Dataset

class SmilesMLMDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_length=128):
        self.smiles = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.smiles[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
