import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 遍历文件夹，加载图像路径和标签
        for label_dir in sorted(os.listdir(root_dir)):
            for image_name in os.listdir(os.path.join(root_dir, label_dir)):
                self.images.append(os.path.join(root_dir, label_dir, image_name))
                self.labels.append(int(label_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]

        # 加载图像
        image = Image.open(image_path).convert('L')  # 转换为灰度图

        # 背景归一化
        image = ImageOps.invert(image)  # 反转图像颜色
        image = ImageOps.equalize(image)  # 直方图均衡化

        # 应用预处理
        if self.transform:
            image = self.transform(image)

        return image, label


# 数据预处理和数据增强
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整图像大小
    transforms.RandomRotation(10),  # 随机旋转
    transforms.RandomAffine(0, shear=10),  # 随机剪切
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 随机调整亮度、对比度和饱和度
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 创建自定义数据集
train_dataset = CustomDataset(root_dir=r'C:\\Users\\DELL\\Desktop\\numtext', transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

# 定义卷积神经网络 (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 第一卷积层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 第二卷积层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 全连接层
        self.fc2 = nn.Linear(128, 10)  # 输出层

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # 第一卷积层激活
        x = self.pool(x)  # 池化
        x = torch.relu(self.conv2(x))  # 第二卷积层激活
        x = self.pool(x)  # 池化
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = torch.relu(self.fc1(x))  # 全连接层激活
        x = self.fc2(x)  # 输出层
        return x

# 实例化模型并移动到 GPU
model = CNN().to(device)

# 如果有多块 GPU，使用 DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# 定义 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

# 定义损失函数和优化器
criterion = FocalLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# 训练模型并保存结果
def train_model(model, train_loader, criterion, optimizer, epochs):
    loss_values = []
    accuracy_values = []

    for epoch in range(epochs):
        running_loss = 0
        correct = 0
        total = 0
        model.train()  # 设置模型为训练模式

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据移动到 GPU

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        loss_values.append(epoch_loss)
        accuracy_values.append(epoch_accuracy)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

    return loss_values, accuracy_values

# 主程序入口
if __name__ == '__main__':
    # 训练模型
    loss_values, accuracy_values = train_model(model, train_loader, criterion, optimizer, epochs=50)

    # 保存训练好的模型
    torch.save(model.state_dict(), 'handwritten_digit_model_1.pth')
