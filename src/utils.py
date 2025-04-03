import h5py
import torch
from torch.utils.data import Dataset
from datetime import datetime
import uuid

class HDF5Dataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            self.images = f["images"][:]
            self.labels = f["labels"][:]
        self.images = torch.tensor(self.images, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def generate_experiment_id():
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"exp-{now}"