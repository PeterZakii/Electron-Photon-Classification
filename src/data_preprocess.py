import os
from pathlib import Path
import h5py
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(photon_file, electron_file):

    with h5py.File(photon_file, "r") as f:
        photons = f["X"][:]  # shape: (N1, H, W, 2)
    with h5py.File(electron_file, "r") as f:
        electrons = f["X"][:]  # shape: (N2, H, W, 2)
    return photons, electrons

def to_channel_first(x):
    """
    transpose the data to channel_first format since GPU operations are optimized on it.  
    (N, H, W, C) -> (N, C, H, W)
    """
    return np.transpose(x, (0, 3, 1, 2))  

def normalize_data(x, mean, std):
    return (x - mean[:, None, None]) / std[:, None, None] 

def save_hdf5(filepath, X, y):
    with h5py.File(filepath, "w") as f:
        f.create_dataset("images", data=X, compression="gzip")
        f.create_dataset("labels", data=y.astype(np.int64))


def main():
    photon_file = 'SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5'
    electron_file = 'SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5'

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / 'data'

    photon_path = data_dir / photon_file
    electron_path = data_dir / electron_file
    
    photons, electrons = load_data(photon_path, electron_path)

    photon_labels = np.zeros(len(photons), dtype=np.int64)
    electron_labels = np.ones(len(electrons), dtype=np.int64)

    data = np.concatenate([photons, electrons], axis=0)
    labels = np.concatenate([photon_labels, electron_labels], axis=0)

    ## not needed since train_test_split shuffles by default
    # combined = list(zip(data, labels))
    # np.random.shuffle(combined)
    # data, lables = zip(*combined)
    # data = np.array(data)
    # lables = np.array(lables)

    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

    X_train = to_channel_first(X_train)
    X_val = to_channel_first(X_val)
    X_test = to_channel_first(X_test)

    # calculate the mean and std per channel (channel_first format)
    mean = X_train.mean(axis=(0, 2, 3)) 
    std = X_train.std(axis=(0, 2, 3))

    X_train = normalize_data(X_train, mean, std)
    X_val = normalize_data(X_val, mean, std)
    X_test = normalize_data(X_test, mean, std)

    
    preprocessed_data_dir = base_dir.parent / 'data/preprocessed'

    train_dataset_dir = preprocessed_data_dir / 'train_dataset.hdf5'
    validation_dataset_dir = preprocessed_data_dir / "val_dataset.hdf5"
    test_dataset_dir = preprocessed_data_dir / "test_dataset.hdf5"

    save_hdf5(train_dataset_dir, X_train, y_train)
    save_hdf5(validation_dataset_dir, X_val, y_val)
    save_hdf5(test_dataset_dir, X_test, y_test)

    # save mean and std for later use in inference
    np.savez("normalization_stats.npz", mean=mean, std=std)

if __name__ == "__main__":
    main()