% 读取训练结果数据
filename = 'training_results.txt';
fileID = fopen(filename, 'r');
data = textscan(fileID, 'Epoch %d, Loss: %f, Accuracy: %f');
fclose(fileID);

% 提取Epoch, Loss和Accuracy数据
epochs = data{1};
losses = data{2};
accuracies = data{3};

% 创建图形窗口
figure;

% 绘制损失曲线
subplot(2, 1, 1); % 两行一列的第一个子图
plot(epochs, losses, '-b', 'LineWidth', 2);
xlabel('Epoch');
ylabel('Loss');
title('Training Loss');
grid on;

% 绘制准确率曲线
subplot(2, 1, 2); % 两行一列的第二个子图
plot(epochs, accuracies, '-r', 'LineWidth', 2);
xlabel('Epoch');
ylabel('Accuracy');
title('Training Accuracy');
grid on;

% 保存图形为PNG文件
saveas(gcf, 'training_results.png');