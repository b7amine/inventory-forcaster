import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, seq_inputs, scalar_inputs, targets, target_weeks):
        self.seq_inputs = seq_inputs
        self.scalar_inputs = scalar_inputs
        self.targets = targets
        self.target_weeks = target_weeks
    
    def __len__(self):
        return len(self.seq_inputs)
    
    def __getitem__(self, idx):
        return self.seq_inputs[idx], self.scalar_inputs[idx], self.targets[idx], self.target_weeks[idx]