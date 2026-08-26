import torch
import torch.nn as nn
import pennylane as qml
import numpy as np

# 4-qubit device for trainable variational circuit
n_qubits = 4
try:
    dev = qml.device("lightning.qubit", wires=n_qubits)
except ImportError:
    dev = qml.device("default.qubit", wires=n_qubits)


def make_qnode(n_layers=1):
    """
    Creates a differentiable QNode with AngleEmbedding and BasicEntanglerLayers.
    Supports native batching and parameter-shift rule.
    """
    @qml.qnode(dev, diff_method="parameter-shift", interface="torch")
    def qnode(inputs, weights):
        # 1. Batched angle encoding: pixel values in [0, 1] mapped via RY(pi * x)
        qml.AngleEmbedding(inputs * np.pi, wires=list(range(n_qubits)), rotation='Y')
        
        # 2. Trainable Variational Layers: weights shape (n_layers, n_qubits)
        qml.BasicEntanglerLayers(weights=weights, wires=list(range(n_qubits)))
        
        # 3. Pauli-Z expectation values on all 4 qubits
        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]
    
    return qnode


class TrainableQuanvLayer(nn.Module):
    """
    PyTorch Module that slides a trainable 4-qubit quantum kernel (2x2 patch, stride 2)
    over an input image tensor of shape (B, 1, 28, 28) -> produces (B, 4, 14, 14).
    """
    def __init__(self, n_layers=1):
        super(TrainableQuanvLayer, self).__init__()
        self.n_layers = n_layers
        self.qnode = make_qnode(n_layers=n_layers)
        
        weight_shapes = {"weights": (n_layers, n_qubits)}
        # Initialize quantum weights with uniform random [0, 2*pi]
        init_weights = {"weights": lambda x: torch.rand(n_layers, n_qubits) * 2 * np.pi}
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes, init_method=init_weights)

    def forward(self, x):
        """
        x: (B, 1, 28, 28) in [0, 1] range.
        Output: (B, 4, 14, 14) feature maps.
        """
        B = x.shape[0]
        # Unfold 2x2 patches with stride 2: (B, 1, 14, 14, 2, 2)
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
        # Reshape to (B * 14 * 14, 4) -> 196 patches per image
        patches = patches.contiguous().view(B, 14, 14, 4).view(-1, 4)
        
        # Pass batch of 4-pixel patches through differentiable quantum layer
        q_out = self.qlayer(patches)  # (B * 196, 4)
        
        # Reshape back to (B, 4, 14, 14)
        q_out = q_out.view(B, 14, 14, 4).permute(0, 3, 1, 2).contiguous()
        return q_out


class TrainableQuanvolutionalNetwork(nn.Module):
    """
    Full End-to-End Hybrid Quantum-Classical Network with Trainable Quantum Kernel:
      Image (1, 28, 28)
         │
      [TrainableQuanvLayer] -> (4, 14, 14)
         │
      [BatchNorm2d(4)]
         │
      [ReLU()]
         │
      [Flatten()] -> (784,)
         │
      [Linear(784, num_classes)]
    """
    def __init__(self, num_classes=4, n_layers=1):
        super(TrainableQuanvolutionalNetwork, self).__init__()
        self.quanv = TrainableQuanvLayer(n_layers=n_layers)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(784, num_classes)

    def forward(self, x):
        x = self.quanv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def get_parameter_groups(self, lr_quantum=0.01, lr_classical=0.001):
        """
        Separates quantum parameters (for higher lr) and classical parameters.
        """
        quantum_params = list(self.quanv.parameters())
        classical_params = list(self.bn.parameters()) + list(self.fc.parameters())
        
        return [
            {'params': quantum_params,   'lr': lr_quantum,   'name': 'quantum'},
            {'params': classical_params, 'lr': lr_classical, 'name': 'classical'}
        ]
