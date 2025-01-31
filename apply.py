import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# 定义模型结构（与训练时的模型结构一致）
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

# 加载模型
model = NeuralNet()
model.load_state_dict(torch.load('handwritten_digit_model.pth'))
model.eval()  # 设置为评估模式

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # 调整图像大小
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 定义预测函数
def predict_image(image_path):
    # 加载图像
    image = Image.open(image_path).convert('L')  # 转换为灰度图
    image = transform(image).unsqueeze(0)  # 添加批次维度

    # 预测
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# 创建GUI
class DigitPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字预测")

        # 创建一个标签用于显示图像
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=20)

        # 创建一个按钮用于选择文件
        self.load_button = tk.Button(root, text="选择图像", command=self.load_image)
        self.load_button.pack(pady=10)

        # 创建一个标签用于显示预测结果
        self.result_label = tk.Label(root, text="", font=("Arial", 16))
        self.result_label.pack(pady=20)

    def load_image(self):
        # 打开文件选择对话框
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            self.predict_and_display(file_path)

    def predict_and_display(self, image_path):
        # 预测数字
        predicted_digit = predict_image(image_path)
        self.result_label.config(text=f"预测的数字是: {predicted_digit}")

        # 显示图像
        image = Image.open(image_path)
        image = image.resize((200, 200), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo  # 防止被垃圾回收

# 主程序
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitPredictorApp(root)
    root.mainloop()