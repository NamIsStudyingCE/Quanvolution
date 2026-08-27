import os
import torch
import numpy as np
import pennylane as qml
from tqdm import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed

# 4-qubit Quanvolution circuit (2x2 patch) — Henderson (2019)
n_qubits = 4
try:
    dev = qml.device("lightning.qubit", wires=n_qubits)
except ImportError:
    dev = qml.device("default.qubit", wires=n_qubits)

# Fixed random parameters — seed 42 matches the data split seed for full reproducibility
np.random.seed(42)
rand_params = np.random.uniform(high=2 * np.pi, size=(n_qubits,))


@qml.qnode(dev)
def circuit(phi):
    """
    Encode 4 pixels as RY rotations, apply fixed random quantum layer, measure PauliZ.
    phi must be in [0, 1] so RY(pi * phi) sweeps [0, pi] (|0> to |1>).
    Using normalized pixels [-1, 1] would extend rotations to [-pi, pi] and
    break the encoding intent of Henderson (2019).
    """
    for j in range(n_qubits):
        qml.RY(np.pi * phi[j], wires=j)

    qml.RandomLayers(weights=[rand_params], wires=list(range(n_qubits)))

    return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]


def quanv(image):
    """
    Apply Quanvolution to a single image.
    image shape: (1, 28, 28), pixel values in [0, 1].
    Returns: (4, 14, 14) feature map.
    """
    out = np.zeros((4, 14, 14))
    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            q_results = circuit([
                image[0, j,   k],
                image[0, j,   k+1],
                image[0, j+1, k],
                image[0, j+1, k+1]
            ])
            for c in range(4):
                out[c, j//2, k//2] = q_results[c]
    return out


def precompute_split(dataset, split_name, save_path):
    if os.path.exists(save_path):
        print(f"Skip {split_name}. File exists: {save_path}")
        return

    print(f"Precompute quantum features for {split_name}...")

    # dataset here is the quantum_* split (ToTensor only, pixels in [0, 1])
    images = [dataset[i][0].numpy() for i in range(len(dataset))]
    labels = [dataset[i][1] for i in range(len(dataset))]

    num_cores = mp.cpu_count()

    features = Parallel(n_jobs=num_cores)(
        delayed(quanv)(img) for img in tqdm(images, desc=f"Processing {split_name}")
    )

    features_tensor = torch.tensor(np.array(features), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    torch.save((features_tensor, labels_tensor), save_path)
    print(f"Saved {features_tensor.shape} to {save_path}")


def run_precompute(dataset_name="breastmnist", data_splits=None, save_dir="data/quantum_features"):
    """
    Precompute quantum features from the quantum_* splits (pixels in [0, 1]).
    Skips splits that already have cached .pt files.
    """
    os.makedirs(save_dir, exist_ok=True)

    for split in ['train', 'val', 'test']:
        dataset = data_splits[f'quantum_{split}']
        save_path = os.path.join(save_dir, f"{dataset_name}_{split}.pt")
        precompute_split(dataset, split, save_path)
