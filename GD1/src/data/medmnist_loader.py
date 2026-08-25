import os
import torch
import numpy as np
from medmnist import BreastMNIST, OCTMNIST
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def load_and_subset_medmnist(dataset_class, split, transform, max_samples=None, seed=42):
    dataset = dataset_class(split=split, download=True, size=28)

    # Extract raw PIL images and scalar labels
    images = [dataset[i][0] for i in range(len(dataset))]
    labels = [dataset[i][1][0] for i in range(len(dataset))]

    if max_samples is not None and len(labels) > max_samples:
        np.random.seed(seed)
        _, subset_indices = train_test_split(
            np.arange(len(labels)),
            test_size=max_samples,
            stratify=labels,
            random_state=seed
        )
        images = [images[i] for i in subset_indices]
        labels = [labels[i] for i in subset_indices]

    return CustomDataset(images, labels, transform=transform)


def prepare_data(dataset_name="breastmnist", max_train=None, max_val=None, max_test=None, seed=42, save_dir="data/processed"):
    """
    Prepare and cache deterministic train/val/test splits.

    Stores two transform variants per split:
      - classical: ToTensor + Normalize(0.5, 0.5) → pixel range [-1, 1]
      - quantum_raw: ToTensor only              → pixel range [0, 1]

    Quantum encoding RY(pi * phi) is designed for phi in [0, 1] (Henderson 2019).
    Classical CNN benefits from standard [-1, 1] normalization.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset_name}_splits.pt")

    if os.path.exists(save_path):
        print(f"Load existing splits from {save_path}")
        return torch.load(save_path, weights_only=False)

    # Classical transform: normalized to [-1, 1]
    classical_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Quantum transform: only ToTensor, pixel in [0, 1] as required by RY(pi * phi)
    quantum_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if dataset_name == "breastmnist":
        dataset_class = BreastMNIST
    elif dataset_name == "octmnist":
        dataset_class = OCTMNIST
    else:
        raise ValueError("Unsupported dataset")

    # Classical datasets (normalized images for CNN)
    c_train = load_and_subset_medmnist(dataset_class, 'train', classical_transform, max_train, seed)
    c_val   = load_and_subset_medmnist(dataset_class, 'val',   classical_transform, max_val,   seed)
    c_test  = load_and_subset_medmnist(dataset_class, 'test',  classical_transform, max_test,  seed)

    # Quantum datasets (raw [0,1] images for quantum encoding)
    q_train = load_and_subset_medmnist(dataset_class, 'train', quantum_transform, max_train, seed)
    q_val   = load_and_subset_medmnist(dataset_class, 'val',   quantum_transform, max_val,   seed)
    q_test  = load_and_subset_medmnist(dataset_class, 'test',  quantum_transform, max_test,  seed)

    splits = {
        'classical_train': c_train,
        'classical_val':   c_val,
        'classical_test':  c_test,
        'quantum_train':   q_train,
        'quantum_val':     q_val,
        'quantum_test':    q_test,
        'num_classes': len(np.unique(c_train.labels))
    }

    torch.save(splits, save_path)
    print(f"Save splits to {save_path}")
    return splits
