import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def reshape_data(x):
    batch_size = x.size(0)
    x = x.view(batch_size, 1, 10, 47)
    return x

class ReshapeDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x.values
        self.y = y.values
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        features = torch.tensor(self.x[index], dtype=torch.float32)
        labels = torch.tensor(self.y[index], dtype=torch.long)
        
        features = features.view(1, 10, 47)
        
        return features, labels
    
    