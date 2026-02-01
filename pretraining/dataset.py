from torch.utils.data import Dataset

class SmilesMLMDataset(Dataset):
    def __init__(self, smiles_list):
        self.smiles = smiles_list

    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        return self.smiles[idx]