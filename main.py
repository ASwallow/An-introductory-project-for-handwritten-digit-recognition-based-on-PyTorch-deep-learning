import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import lxml

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

        # 应用预处理
        if self.transform:
            image = self.transform(image)

        return image, label


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整图像大小
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 创建自定义数据集
train_dataset = CustomDataset(root_dir=r'numtext', transform=transform)#读取图片，记得更改地址

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# 定义模型
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 64)  # 隐藏层到隐藏层
        self.fc3 = nn.Linear(64, 10)  # 隐藏层到输出层

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 展平图像
        x = torch.relu(self.fc1(x))  # 第一个隐藏层
        x = torch.relu(self.fc2(x))  # 第二个隐藏层
        x = self.fc3(x)  # 输出层
        return x


# 实例化模型
model = NeuralNet()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型并保存结果
def train_model(model, train_loader, criterion, optimizer, epochs):
    loss_values = []
    accuracy_values = []
    for epoch in range(epochs):
        running_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

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


# 训练模型
loss_values, accuracy_values = train_model(model, train_loader, criterion, optimizer, epochs=50)

# 保存训练结果到TXT文件
with open('training_results.txt', 'w') as f:
    for epoch, loss, accuracy in zip(range(1, 51), loss_values, accuracy_values):
        f.write(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n")