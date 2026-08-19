import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ==========================================
# 1. Cấu hình và Siêu tham số (Hyperparameters)
# ==========================================
n_epochs = 5      # Số lượng epoch huấn luyện
n_layers = 1      # Số lớp ngẫu nhiên trong Quantum Circuit
n_train = 50      # Kích thước tập huấn luyện (nhỏ để demo nhanh)
n_test = 30       # Kích thước tập kiểm thử

# Đặt seed để dễ dàng tái lập kết quả (reproducibility)
np.random.seed(0)
torch.manual_seed(0)

# ==========================================
# 2. Chuẩn bị dữ liệu (Dataset: MNIST)
# ==========================================
print("Đang tải dữ liệu MNIST...")
transform = transforms.Compose([
    transforms.ToTensor(), # Chuyển ảnh PIL thành PyTorch Tensor (scale [0, 1])
])

# Tải tập dữ liệu MNIST từ torchvision
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Trích xuất một subset nhỏ để demo chạy nhanh
train_dataset = Subset(train_dataset, range(n_train))
test_dataset = Subset(test_dataset, range(n_test))

# Sử dụng DataLoader, batch_size=1 vì xử lý Quantum filter thủ công từng ảnh
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ==========================================
# 3. Định nghĩa Quantum Circuit (Quanvolution Layer)
# ==========================================
# Khởi tạo quantum device với 4 qubits (tương ứng với 4 pixel của kernel 2x2)
dev = qml.device("default.qubit", wires=4)

# Khởi tạo các tham số ngẫu nhiên cho lớp lượng tử (random layers)
rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))

@qml.qnode(dev, interface="torch")
def circuit(phi):
    """
    Quantum circuit nhận 4 giá trị pixel đầu vào, mã hóa, cho đi qua 
    tương tác ngẫu nhiên và đo lường.
    """
    # 3.1. Mã hóa giá trị pixel (đã được scale) vào góc quay RY
    for j in range(4):
        qml.RY(np.pi * phi[j], wires=j)

    # 3.2. Lớp tương tác/vướng víu (Entanglement Layer)
    # Sử dụng qml.RandomLayers làm ví dụ cho tương tác cố định/ngẫu nhiên
    qml.RandomLayers(rand_params, wires=list(range(4)))

    # 3.3. Phép đo (Measurement)
    # Đo giá trị kỳ vọng (expectation value) của toán tử Pauli-Z trên cả 4 qubits
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

# Vẽ circuit lượng tử
print("\nKiến trúc Quantum Circuit:")
print(qml.draw(circuit)(np.array([0.1, 0.2, 0.3, 0.4])))

# ==========================================
# 4. Quanvolution Filter (Hàm Tiền Xử Lý)
# ==========================================
def quanv(image):
    """
    Áp dụng phép tích chập lượng tử (Quanvolution) lên ảnh đầu vào.
    Sử dụng kernel kích thước 2x2 và stride = 2.
    
    Args:
        image (torch.Tensor): Ảnh có shape (1, 28, 28)
        
    Returns:
        torch.Tensor: Đặc trưng lượng tử (4 channels), shape (4, 14, 14)
    """
    # Khởi tạo tensor output (4 kênh đặc trưng, kích thước 14x14 do stride=2)
    out = torch.zeros((4, 14, 14))

    # Trượt kernel kích thước 2x2 trên ảnh với stride = 2
    for j in range(0, 28, 2):
        for k in range(0, 28, 2):
            # Lấy 4 pixel từ không gian ảnh tương ứng với kernel hiện tại
            # (Ảnh grayscale nên channel color = 0)
            q_results = circuit(
                [
                    image[0, j, k],
                    image[0, j, k + 1],
                    image[0, j + 1, k],
                    image[0, j + 1, k + 1]
                ]
            )
            # Gán kết quả đo vào 4 kênh của output tại vị trí (j//2, k//2)
            for c in range(4):
                out[c, j // 2, k // 2] = q_results[c]
    return out

# Demo trực quan hóa một ảnh qua bộ lọc Quanvolution
sample_img, sample_label = next(iter(train_loader))
sample_img = sample_img[0] # Lấy ảnh từ batch, kích thước (1, 28, 28)

print(f"\nShape ảnh gốc trước Quanvolution: {sample_img.shape}")
q_features = quanv(sample_img)
print(f"Shape đặc trưng sau Quanvolution: {q_features.shape}")

# Trực quan hóa (Matplotlib) so sánh ảnh gốc và 4 feature maps
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
axes[0].imshow(sample_img[0].numpy(), cmap='gray')
axes[0].set_title("Ảnh gốc (28x28)")
axes[0].axis('off')

for i in range(4):
    axes[i+1].imshow(q_features[i].detach().numpy(), cmap='gray')
    axes[i+1].set_title(f"Q-Channel {i} (14x14)")
    axes[i+1].axis('off')

plt.tight_layout()
plt.savefig("quanvolution_features.png")
print("\nĐã lưu ảnh trực quan hóa đặc trưng tại 'quanvolution_features.png'.")

# ==========================================
# 5. Tiền xử lý toàn bộ tập dữ liệu (Classical Pipeline)
# ==========================================
print("\nBắt đầu tiền xử lý dữ liệu (Feature Extraction)...")

# Trích xuất đặc trưng lượng tử cho Train set
q_train_images = []
print(f"Đang xử lý Train set ({n_train} ảnh)...")
for idx, (img, label) in enumerate(train_loader):
    q_train_images.append(quanv(img[0]))
q_train_images = torch.stack(q_train_images) # Stack thành Tensor (50, 4, 14, 14)
train_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])

# Trích xuất đặc trưng lượng tử cho Test set
q_test_images = []
print(f"Đang xử lý Test set ({n_test} ảnh)...")
for idx, (img, label) in enumerate(test_loader):
    q_test_images.append(quanv(img[0]))
q_test_images = torch.stack(q_test_images) # Stack thành Tensor (30, 4, 14, 14)
test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

print("Đã hoàn thành tiền xử lý!")
print(f"Train features shape (Tổng quát): {q_train_images.shape}")

# ==========================================
# 6. Mạng nơ-ron phân loại cổ điển (Linear Classifier)
# ==========================================
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        # Flattened size = 4 channels * 14 * 14 = 784
        # Đầu ra 10 classes cho bài toán phân loại số 0-9
        self.fc = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x):
        # Flatten tensor từ (Batch, 4, 14, 14) -> (Batch, 784)
        x = x.view(x.shape[0], -1) 
        x = self.fc(x)
        return x

model = SimpleClassifier()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# ==========================================
# 7. Huấn luyện mô hình (Training Pipeline)
# ==========================================
print("\nBắt đầu huấn luyện mô hình cổ điển trên các Quantum Features...")
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    
    # Do dữ liệu đã được tiền xử lý thành batch lớn, ta train trực tiếp 
    # (Để đơn giản demo, train từng mẫu, có thể dùng DataLoader batch lớn hơn)
    for i in range(n_train):
        optimizer.zero_grad()
        
        # Thêm chiều batch_size=1
        out = model(q_train_images[i].unsqueeze(0)) 
        loss = criterion(out, train_labels[i].unsqueeze(0))
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    # Đánh giá trên tập test (Evaluate)
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(n_test):
            out = model(q_test_images[i].unsqueeze(0))
            pred = out.argmax(dim=1)
            if pred == test_labels[i]:
                correct += 1
                
    accuracy = correct / n_test
    avg_loss = running_loss / n_train
    print(f"Epoch {epoch+1:02d}/{n_epochs} | Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.4f}")

print("\nHoàn tất chạy demo Quanvolutional Neural Network!")
