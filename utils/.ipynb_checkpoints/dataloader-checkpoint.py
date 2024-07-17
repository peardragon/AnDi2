from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import re


class AnDiDataset(Dataset):
    def __init__(self, data, labels):
        """
        data: List of tensors
        labels: List of nested lists (labels can be of varying lengths or structures)
        """
        self.data = data
        self.labels = labels
        self.indices = len(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.indices:
            raise IndexError(f"Index {idx} is out of range for dataset of size {len(self.indices)}")

        return self.data[idx], self.labels[idx]

def collate_fn(batch):
    """
    Custom collate function for handling batches where labels are nested lists.
    
    Parameters:
    - batch: List of tuples (data, label)
    
    Returns:
    - data_tensor: Tensor stacked from data
    - labels: List of labels (unchanged structure)
    """
    # Separate data and labels
    data = [item[0] for item in batch]
    label = [item[1] for item in batch]
    
    # Stack data into a single tensor
    data_tensor = torch.stack(data, dim=0)
    label_tensor = torch.stack(label, dim=0)
    
    return data_tensor, label_tensor