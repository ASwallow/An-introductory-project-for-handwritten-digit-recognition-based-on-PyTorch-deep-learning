import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps

import tkinter as tk
from tkinter import filedialog, messagebox



# 加载模型
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

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


# 加载和预处理图像
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path)
    image = ImageOps.invert(image)  # 反转颜色
    image = ImageOps.equalize(image)  # 直方图均衡化
    image = transform(image).unsqueeze(0)  # 添加批次维度
    return image


# 预测函数
def predict(model, device, image_path):
    image = preprocess_image(image_path)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence


# 图形界面部分
def drag_and_drop_window(model, device):
    def open_file():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.bmp")])
        if file_path:
            try:
                predicted_class, confidence = predict(model, device, file_path)
                messagebox.showinfo("预测结果", f"预测值: {predicted_class}\n预测概率: {confidence:.4f}")
            except Exception as e:
                messagebox.showerror("错误", f"预测出错: {str(e)}")

    # 创建窗口
    window = tk.Tk()
    window.title("手写数字识别")
    window.geometry("400x200")
    window.configure(bg="white")

    # 添加文件选择按钮
    button = tk.Button(window, text="选择图片", command=open_file, font=("Arial", 16))
    button.pack(pady=50)

    window.mainloop()


# 主程序
if __name__ == '__main__':
    # 加载模型
    model_path = 'handwritten_digit_model_1.pth'
    model, device = load_model(model_path)

    # 启动选择文件窗口
    drag_and_drop_window(model, device)