"""Forensic audit Task 2+3: instantiate all models, count parameters,
and verify backprop vs parameter-shift gradients."""
import sys
sys.path.insert(0, '.')
import numpy as np
import torch
import pennylane as qml

print("=" * 80)
print("TASK 2: MODEL PARAMETER COUNT (actual instantiation)")
print("=" * 80)
from src.models.classical_cnn import SymmetricalMinimumCNN
from src.models.trainable_quanv import TrainableQuanvolutionalNetwork
from src.models.quantum_model import QuanvolutionClassifier

def count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def breakdown(m):
    out = {}
    for name, p in m.named_parameters():
        out.setdefault(name.split('.')[0], 0)
        out[name.split('.')[0]] += p.numel()
    return out

for K, tag in [(2, 'Breast K=2'), (4, 'OCT K=4')]:
    cnn = SymmetricalMinimumCNN(num_classes=K)
    b = breakdown(cnn)
    print(f"ClassicalCNN {tag}: total={count(cnn)} conv={b.get('conv')} bn={b.get('bn')} fc={b.get('fc')}")

for K, L, ans, tag in [(2, 2, 'basic', 'Breast'), (4, 1, 'basic', 'OCT'),
                        (2, 2, 'strongly', 'Breast'), (4, 1, 'strongly', 'OCT')]:
    m = TrainableQuanvolutionalNetwork(num_classes=K, n_layers=L, ansatz=ans)
    b = breakdown(m)
    print(f"Trainable-{ans}-L{L} {tag} K={K}: total={count(m)} quanv={b.get('quanv')} bn={b.get('bn')} fc={b.get('fc')}")

for K, tag in [(2, 'Breast'), (4, 'OCT')]:
    qm = QuanvolutionClassifier(num_classes=K)
    b = breakdown(qm)
    print(f"FixedQuanvClassifier {tag} K={K}: total={count(qm)} bn={b.get('bn')} fc={b.get('fc')}")

print()
print("=" * 80)
print("TASK 3: BACKPROP vs PARAMETER-SHIFT VERIFICATION (independent rerun)")
print("=" * 80)
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

torch.manual_seed(0)
np.random.seed(0)

@qml.qnode(dev, diff_method="backprop", interface="torch")
def circuit_backprop(inputs, weights):
    qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits), rotation='Y')
    qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

def circuit_shift(inputs, weights_np):
    """Parameter-shift on strongly-entangling Rot gates: d/dtheta of Rot(a,b,c) wrt a,c uses shift pi/2; wrt b uses general two-term rule."""
    # Exact analytic gradient via adjacent-difference on tapes would be complex;
    # use PennyLane's built-in param-shift for ground truth instead.
    @qml.qnode(dev, diff_method="parameter-shift")
    def q(inputs, weights):
        qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits), rotation='Y')
        qml.StronglyEntanglingLayers(weights=weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]
    w = torch.tensor(weights_np, requires_grad=True)
    res = q(inputs, w)
    g = torch.autograd.grad(res, w)[0].numpy()
    return g

x = torch.rand(n_qubits)
w = torch.rand(1, n_qubits, 3) * 2 * np.pi
w_t = w.clone().requires_grad_(True)
res = circuit_backprop(x, w_t)
g_bp = torch.autograd.grad(res, w_t)[0].numpy()
g_ps = circuit_shift(x, w.detach().numpy())
delta = np.abs(g_bp - g_ps)
print(f"  Circuit: StronglyEntangling L1, 4 qubits, 12 params, random patch")
print(f"  max |backprop - parameter-shift| = {delta.max():.3e}")
print(f"  paper claim: |Delta| < 4.1e-8  ->  {'CONSISTENT' if delta.max() < 4.1e-8 else 'CHECK: ' + format(delta.max(), '.3e')}")

# Also test L2 basic
@qml.qnode(dev, diff_method="backprop", interface="torch")
def circ_basic_bp(inputs, weights):
    qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits), rotation='Y')
    qml.BasicEntanglerLayers(weights=weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

@qml.qnode(dev, diff_method="parameter-shift")
def circ_basic_ps(inputs, weights):
    qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits), rotation='Y')
    qml.BasicEntanglerLayers(weights=weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]

w2 = (torch.rand(2, n_qubits) * 2 * np.pi).requires_grad_(True)
r = circ_basic_bp(x, w2)
g1 = torch.autograd.grad(r, w2)[0].numpy()
w2b = w2.detach().requires_grad_(True)
r2 = circ_basic_ps(x, w2b)
g2 = torch.autograd.grad(r2, w2b)[0].numpy()
print(f"  Circuit: BasicEntangler L2, 8 params: max |bp - ps| = {np.abs(g1-g2).max():.3e}")
