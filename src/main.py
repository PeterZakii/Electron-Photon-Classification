from pathlib import Path
import torch
from torch.utils.data import DataLoader, RandomSampler
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from model import ResNet15
from train import Trainer
from utils import HDF5Dataset, generate_experiment_id


def main():
    experiment_id = generate_experiment_id()
    print(f"Experiment num: {experiment_id} started")

    base_dir = Path(__file__).resolve().parent.parent
    prep_data_dir = base_dir / "data" / "preprocessed"

    # Dataset paths
    train_file = prep_data_dir / "train_dataset.hdf5"
    val_file = prep_data_dir / "val_dataset.hdf5"
    test_file = prep_data_dir / "test_dataset.hdf5"

    # Load datasets
    print(f"Loading datasets...")
    train_dataset = HDF5Dataset(train_file)
    val_dataset = HDF5Dataset(val_file)
    test_dataset = HDF5Dataset(test_file)

    # Create DataLoaders
    print(f"Creating DataLoaders...")
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=RandomSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Setup model and trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet15(img_channels=2, num_classes=1)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'max', patience=2, factor=0.5, verbose=True)

    trainer = Trainer(model, criterion, optimizer, scheduler, train_loader, val_loader, test_loader, device, experiment_id, base_dir)

    # Train and test
    trainer.train(num_epochs=35)
    trainer.test()


if __name__ == "__main__":
    main()
