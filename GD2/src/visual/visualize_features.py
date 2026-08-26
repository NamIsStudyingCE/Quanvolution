import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root and src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.medmnist_loader import prepare_data
from data.precompute_features import quanv
from models.classical_cnn import SymmetricalMinimumCNN

# Label mappings for medical datasets
LABEL_MAPS = {
    'breastmnist': {
        0: 'Malignant (Ác tính)',
        1: 'Benign (Lành tính)'
    },
    'octmnist': {
        0: 'CNV (Tân mạch hắc mạc)',
        1: 'DME (Phù hoàng điểm ĐTĐ)',
        2: 'DRUSEN (Thoái hóa hoàng điểm)',
        3: 'NORMAL (Bình thường)'
    }
}

def get_sample_indices_per_class(dataset, num_classes):
    """Find the first sample index for each class in the dataset."""
    class_indices = {}
    for idx in range(len(dataset)):
        label = dataset[idx][1]
        if isinstance(label, np.ndarray) or isinstance(label, torch.Tensor):
            label = int(label)
        if label not in class_indices:
            class_indices[label] = idx
        if len(class_indices) == num_classes:
            break
    return class_indices

def visualize_dataset_features(dataset_name="breastmnist", save_dir="results/figures"):
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n>>> Extracting & Visualizing Feature Maps for {dataset_name.upper()}...")
    
    # 1. Load data splits
    splits = prepare_data(dataset_name=dataset_name, max_train=500, max_val=100, max_test=100, seed=42)
    num_classes = splits['num_classes']
    
    c_data = splits['classical_test']
    q_data = splits['quantum_test']
    
    # 2. Get samples for each class
    sample_indices = get_sample_indices_per_class(q_data, num_classes)
    
    # Initialize classical conv filter (fixed seed for deterministic weights)
    torch.manual_seed(42)
    classical_model = SymmetricalMinimumCNN(num_classes=num_classes)
    classical_model.eval()
    
    # 3. Create subplots: Rows = num_classes, Cols = 1 (Original) + 4 (Quantum) + 4 (Classical) = 9 columns
    fig, axes = plt.subplots(num_classes, 9, figsize=(20, 2.5 * num_classes + 1))
    if num_classes == 1:
        axes = np.expand_dims(axes, 0)
        
    for row_idx, (class_label, sample_idx) in enumerate(sorted(sample_indices.items())):
        class_name = LABEL_MAPS.get(dataset_name, {}).get(class_label, f"Class {class_label}")
        
        # Get raw image & tensor
        q_img_tensor, _ = q_data[sample_idx]  # [0, 1] range, shape: (1, 28, 28)
        c_img_tensor, _ = c_data[sample_idx]  # [-1, 1] range, shape: (1, 28, 28)
        
        # Quantum Feature Extraction (4 channels x 14 x 14)
        q_features = quanv(q_img_tensor.numpy())
        
        # Classical Feature Extraction (4 channels x 14 x 14)
        with torch.no_grad():
            c_input = c_img_tensor.unsqueeze(0)  # (1, 1, 28, 28)
            c_features = classical_model.conv(c_input).squeeze(0).cpu().numpy()  # (4, 14, 14)
            
        # Col 0: Original Image (28x28)
        ax_orig = axes[row_idx, 0]
        ax_orig.imshow(q_img_tensor.squeeze(0).numpy(), cmap='gray', vmin=0, vmax=1)
        ax_orig.set_title(f"Input: {class_name}\n[28x28]", fontsize=9, fontweight='bold', pad=6)
        ax_orig.axis('off')
        
        # Cols 1-4: Quantum Feature Maps (14x14)
        for ch in range(4):
            ax_q = axes[row_idx, 1 + ch]
            im_q = ax_q.imshow(q_features[ch], cmap='viridis', vmin=-1, vmax=1)
            if row_idx == 0:
                ax_q.set_title(f"Quantum Q{ch}\n[14x14]", fontsize=9, fontweight='bold', color='#1f77b4', pad=6)
            ax_q.axis('off')
            
        # Cols 5-8: Classical Conv2D Feature Maps (14x14)
        for ch in range(4):
            ax_c = axes[row_idx, 5 + ch]
            im_c = ax_c.imshow(c_features[ch], cmap='magma')
            if row_idx == 0:
                ax_c.set_title(f"Classical C{ch}\n[14x14]", fontsize=9, fontweight='bold', color='#d62728', pad=6)
            ax_c.axis('off')

    plt.suptitle(f"Feature Map Representation Analysis: {dataset_name.upper()}\nOriginal (28x28) vs Quantum Quanvolution (14x14) vs Classical Conv2D (14x14)",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    out_file = os.path.join(save_dir, f"{dataset_name}_feature_comparison.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature comparison visualization to: {out_file}")

def main():
    visualize_dataset_features("breastmnist")
    visualize_dataset_features("octmnist")
    print("\n>>> Feature maps extraction & visualization completed successfully!")

if __name__ == "__main__":
    main()
