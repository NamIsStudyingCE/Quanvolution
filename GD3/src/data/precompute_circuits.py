import os
import torch
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed

from models.circuits import CIRCUIT_DICT, apply_quanv_to_image

def precompute_circuit_split(dataset, split_name, circuit_name, save_path):
    if os.path.exists(save_path):
        return

    circuit_fn = CIRCUIT_DICT[circuit_name]
    
    # Dataset has ToTensor-only transform (pixel range [0, 1])
    images = [dataset[i][0].numpy() for i in range(len(dataset))]
    labels = [dataset[i][1] for i in range(len(dataset))]

    num_cores = mp.cpu_count()

    features = Parallel(n_jobs=num_cores)(
        delayed(apply_quanv_to_image)(img, circuit_fn)
        for img in tqdm(images, desc=f"[{circuit_name}] {split_name}", leave=False)
    )

    features_tensor = torch.tensor(np.array(features), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    torch.save((features_tensor, labels_tensor), save_path)

def precompute_all_circuits_for_dataset(dataset_name, data_splits, save_dir="data/quantum_features"):
    os.makedirs(save_dir, exist_ok=True)
    
    for circuit_name in CIRCUIT_DICT.keys():
        print(f"\n--- Checking/Precomputing Circuit: {circuit_name} on {dataset_name.upper()} ---")
        for split in ['train', 'val', 'test']:
            save_path = os.path.join(save_dir, f"{dataset_name}_{circuit_name}_{split}.pt")
            if os.path.exists(save_path):
                print(f"  [OK] {split} split cached: {save_path}")
            else:
                dataset = data_splits[f'quantum_{split}']
                print(f"  [RUN] Computing {split} split ({len(dataset)} samples)...")
                precompute_circuit_split(dataset, split, circuit_name, save_path)
                print(f"  [SAVED] -> {save_path}")
