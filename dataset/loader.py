from torch.utils.data import Dataset
import numpy as np
import torch
import os

class ZincFingerDataset(Dataset):
    def __init__(self, items, feature_dir):
        self.items = items
        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # Return the feature matrix (dim: 52 x 1,028) and label for the given zinc finger index
        item = self.items[idx]
        feature_path = os.path.join(self.feature_dir, item["npy_path"])
        x = np.load(feature_path)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(item["label"], dtype=torch.float32) # 1 for positive, 0 for negative
        return x, y, item
