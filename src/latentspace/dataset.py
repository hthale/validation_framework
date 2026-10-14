import torch
from torch.utils.data import Dataset

class MultiModalDataset(Dataset):
    def __init__(self, X_tab, X_str_list):
        self.X_tab = torch.tensor(X_tab, dtype=torch.float32)
        self.X_str_list = [torch.tensor(x, dtype=torch.long) for x in X_str_list]

    def __len__(self):
        return len(self.X_tab)

    def __getitem__(self, idx):
        return self.X_tab[idx], [x[idx] for x in self.X_str_list]