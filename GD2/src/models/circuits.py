import numpy as np
import pennylane as qml

n_qubits = 4

# Initialize quantum device
try:
    dev = qml.device("lightning.qubit", wires=n_qubits)
except ImportError:
    dev = qml.device("default.qubit", wires=n_qubits)

# Fixed seed for deterministic weight generation across all ansatzes
np.random.seed(42)

# Weight parameters for different ansatzes
# 1. RandomLayers
random_w_1l = np.random.uniform(high=2 * np.pi, size=(n_qubits,))
random_w_2l = np.random.uniform(high=2 * np.pi, size=(2, n_qubits))

# 2. StronglyEntanglingLayers (Shape: (L, n_qubits, 3))
strongly_w_1l = np.random.uniform(high=2 * np.pi, size=(1, n_qubits, 3))
strongly_w_2l = np.random.uniform(high=2 * np.pi, size=(2, n_qubits, 3))

# 3. BasicEntanglingLayers (Shape: (L, n_qubits))
basic_w_1l = np.random.uniform(high=2 * np.pi, size=(1, n_qubits))
basic_w_2l = np.random.uniform(high=2 * np.pi, size=(2, n_qubits))


# --- QNode Definitions ---

@qml.qnode(dev)
def circuit_random_L1(phi):
    for j in range(n_qubits):
        qml.RY(np.pi * phi[j], wires=j)
    qml.RandomLayers(weights=[random_w_1l], wires=list(range(n_qubits)))
    return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

@qml.qnode(dev)
def circuit_random_L2(phi):
    for j in range(n_qubits):
        qml.RY(np.pi * phi[j], wires=j)
    qml.RandomLayers(weights=random_w_2l, wires=list(range(n_qubits)))
    return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

@qml.qnode(dev)
def circuit_strongly_L1(phi):
    for j in range(n_qubits):
        qml.RY(np.pi * phi[j], wires=j)
    qml.StronglyEntanglingLayers(weights=strongly_w_1l, wires=list(range(n_qubits)))
    return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

@qml.qnode(dev)
def circuit_strongly_L2(phi):
    for j in range(n_qubits):
        qml.RY(np.pi * phi[j], wires=j)
    qml.StronglyEntanglingLayers(weights=strongly_w_2l, wires=list(range(n_qubits)))
    return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

@qml.qnode(dev)
def circuit_basic_L1(phi):
    for j in range(n_qubits):
        qml.RY(np.pi * phi[j], wires=j)
    qml.BasicEntanglerLayers(weights=basic_w_1l, wires=list(range(n_qubits)))
    return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

@qml.qnode(dev)
def circuit_basic_L2(phi):
    for j in range(n_qubits):
        qml.RY(np.pi * phi[j], wires=j)
    qml.BasicEntanglerLayers(weights=basic_w_2l, wires=list(range(n_qubits)))
    return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]


# Dictionary of all 6 circuit ansatzes
CIRCUIT_DICT = {
    'random_L1': circuit_random_L1,
    'random_L2': circuit_random_L2,
    'strongly_L1': circuit_strongly_L1,
    'strongly_L2': circuit_strongly_L2,
    'basic_L1': circuit_basic_L1,
    'basic_L2': circuit_basic_L2,
}

def apply_quanv_to_image(image, circuit_fn):
    """
    Applies a given quantum circuit kernel to a single 1x28x28 image.
    image: numpy array of shape (1, 28, 28) with values in [0, 1].
    Returns: numpy array of shape (4, 14, 14).
    """
    out = np.zeros((4, 14, 14))
    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            q_results = circuit_fn([
                image[0, j,   k],
                image[0, j,   k+1],
                image[0, j+1, k],
                image[0, j+1, k+1]
            ])
            for c in range(4):
                out[c, j//2, k//2] = q_results[c]
    return out
