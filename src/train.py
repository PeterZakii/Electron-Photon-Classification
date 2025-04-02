import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, RandomSampler, DataLoader
from sklearn.metrics import accuracy_score
from model import ResNet15

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
    
base_dir = Path(__file__).resolve().parent
data_dir = base_dir.parent / 'data/preprocessed'

train_data_file = 'train_dataset.hdf5'
val_data_file = 'val_dataset.hdf5'
test_data_file = 'test_dataset.hdf5'

train_dataset = HDF5Dataset(data_dir / train_data_file)
val_dataset = HDF5Dataset(data_dir / val_data_file)
test_dataset = HDF5Dataset(data_dir / test_data_file)

train_sampler = RandomSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)    


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet15(img_channels=2, num_classes=1)
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)



def train(data_loader, model, criterion, optimizer):
    model.train()
    total_loss = 0

    for X, y in data_loader:
        X, y = X.to(device), y.to(device).unsqueeze(1).float()

        y_pred = model(X)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss

def evaluate(data_loader, model):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device).unsqueeze(1).float()

            y_pred = model(X)
            probs = torch.sigmoid(y_pred)
            preds = (probs > 0.5).int()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())

            val_acc = accuracy_score(all_labels, all_preds)

    return val_acc


num_epochs = 10
best_val_acc = 0

for epoch in tqdm(range(num_epochs)):
    # Training
    train_loss = train(train_loader, model, criterion, optimizer)
    
    # Validation
    val_acc = evaluate(val_loader, model)

    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_resnet15.pth")
        print("Model saved.")

    # Print training and validation metrics
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}")


# Testing
print("Testing the model...")
model.load_state_dict(torch.load("best_resnet15.pth"))
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device).unsqueeze(1).float()

        y_pred = model(X)
        probs = torch.sigmoid(y_pred)
        preds = (probs > 0.5).int()

        all_preds.extend(preds.cpu().numpy().flatten())
        all_labels.extend(y.cpu().numpy().flatten())

test_acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {test_acc:.4f}")
