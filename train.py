import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

# Định nghĩa hàm kích hoạt tùy chỉnh
class CustomActivation(nn.Module):
    def forward(self, x):
        return 0.1524 * (x ** 2) + 0.5 * x + 0.409

# Định nghĩa mô hình CNN với ảnh grayscale
class OptimizedCNN(nn.Module):
    def __init__(self):
        super(OptimizedCNN, self).__init__()
        # 3 lớp convolution
        self.conv1 = nn.Conv2d(1, 4, kernel_size=4, padding=0, stride=4)
        
        # Custom activation sau mỗi lớp Conv
        self.custom_activation = CustomActivation()

        # Tính toán kích thước đầu vào cho lớp fully connected
        self.fc_input_size = self._get_fc_input_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 128)  # Lớp fully connected 1
        self.fc2 = nn.Linear(128, 2)  # Lớp fully connected 2 cho 2 nhãn

    def _get_fc_input_size(self):
        # Tính toán kích thước đầu vào cho fully connected layers
        x = torch.zeros(1, 1, 36, 36)  # Tạo một tensor giả với ảnh grayscale
        x = self.conv1(x)
        x = self.custom_activation(x)
        return x.numel()  # Trả về số lượng phần tử trong tensor cuối cùng
        
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.custom_activation(x)

        x = x.view(-1,self.fc_input_size)
        x = self.fc1(x)
        x = self.custom_activation(x)
        x = self.fc2(x)
        return x

# Hàm lưu mô hình sử dụng pickle
def save_model_pickle(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def save_model_state(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model.state_dict(), f)
        
# Hàm huấn luyện mô hình với thanh tiến độ và Early Stopping
def train_model(model, train_loader, criterion, optimizer, scheduler, epochs, patience=5, device='cpu'):
    model.to(device)
    model.train()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Training Epoch [{epoch + 1}/{epochs}]", unit="batch") as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                
                # Đối với CrossEntropyLoss, labels không cần reshape
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Cập nhật thanh tiến độ
                pbar.set_postfix(loss=running_loss / (pbar.n + 1))
                pbar.update(1)

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {avg_loss:.4f}")
        
        scheduler.step(avg_loss)
        
        # Kiểm tra xem có cần dừng không
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            save_model_pickle(model, r'C:\Users\vanvu\Downloads\DoAn\sourcecode\NewFolder\ver5\best_model.pickle')  # Lưu mô hình tốt nhất
            save_model_state(model, r'C:\Users\vanvu\Downloads\DoAn\sourcecode\NewFolder\ver5\best_model_state.pickle')  # Lưu state_dict của mô hình tốt nhất
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping due to no improvement.")
                break

    save_model_pickle(model, r'C:\Users\vanvu\Downloads\DoAn\sourcecode\NewFolder\ver5\final_model.pickle')
    save_model_state(model, r'C:\Users\vanvu\Downloads\DoAn\sourcecode\NewFolder\ver5\final_model_state.pickle')

# Định nghĩa đường dẫn đến các thư mục dữ liệu
base_dir = r'C:\Users\vanvu\Downloads\DoAn\dataset\ABIDE'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Kiểm tra sự tồn tại của các thư mục một lần
if not os.path.exists(train_dir):
    print(f"Directory {train_dir} does not exist.")
if not os.path.exists(validation_dir):
    print(f"Directory {validation_dir} does not exist.")

# Định nghĩa các chuyển đổi cho dữ liệu
train_transform = transforms.Compose([
    transforms.Grayscale(),  # Chuyển đổi ảnh sang grayscale
    transforms.Resize((36, 36)),  # Kích thước ảnh đầu vào
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # Xoay ngẫu nhiên
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.81918], std=[0.24993])
])

# Định nghĩa các chuyển đổi cho dữ liệu validation
validation_transform = transforms.Compose([
    transforms.Grayscale(),  # Chuyển đổi ảnh sang grayscale
    transforms.Resize((36, 36)),  # Kích thước ảnh đầu vào
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.81918], std=[0.24993])
])

# Tải dữ liệu huấn luyện
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=6)

# Tải dữ liệu validation
validation_dataset = datasets.ImageFolder(root=validation_dir, transform=validation_transform)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=6)

# Khởi tạo mô hình, criterion và optimizer
model = OptimizedCNN()  # Khởi tạo mô hình với ảnh grayscale
criterion = nn.CrossEntropyLoss()  # Sử dụng CrossEntropyLoss cho nhiều nhãn
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2)

if __name__ == '__main__':
    # Huấn luyện mô hình
    try:
        train_model(model, train_loader, criterion, optimizer, scheduler, epochs=100, patience=5, device='cpu')
    except Exception as e:
        print(f"An error occurred during training: {e}")

    # Đánh giá mô hình
    def evaluate_model(model, validation_loader, device='cpu'):
        model.eval()
        model.to(device)
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Sử dụng softmax để lấy dự đoán
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()

        avg_loss = total_loss / len(validation_loader)
        accuracy = correct / len(validation_loader.dataset)
        return avg_loss, accuracy

    avg_loss, accuracy = evaluate_model(model, validation_loader, device='cpu')
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
