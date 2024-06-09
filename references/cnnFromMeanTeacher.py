import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F


class CNN(nn.Module):
    """
    CNN from Mean Teacher paper
    """

    def __init__(self, num_classes=10, isL2=False, double_output=False):
        super(CNN, self).__init__()

        self.isL2 = isL2
        self.double_output = double_output

        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = nn.utils.weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = nn.utils.weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = nn.utils.weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1 = nn.Dropout(0.5)

        self.conv2a = nn.utils.weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = nn.utils.weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = nn.utils.weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2 = nn.Dropout(0.5)

        self.conv3a = nn.utils.weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = nn.utils.weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = nn.utils.weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)

        self.fc1 = nn.utils.weight_norm(nn.Linear(128, num_classes))

        if self.double_output:
            self.fc2 = nn.utils.weight_norm(nn.Linear(128, num_classes))

    def forward(self, x, debug=False):

        x = self.activation(self.bn1a(self.conv1a(x)))
        x = self.activation(self.bn1b(self.conv1b(x)))
        x = self.activation(self.bn1c(self.conv1c(x)))
        x = self.mp1(x)
        x = self.drop1(x)

        x = self.activation(self.bn2a(self.conv2a(x)))
        x = self.activation(self.bn2b(self.conv2b(x)))
        x = self.activation(self.bn2c(self.conv2c(x)))
        x = self.mp2(x)
        x = self.drop2(x)

        x = self.activation(self.bn3a(self.conv3a(x)))
        x = self.activation(self.bn3b(self.conv3b(x)))
        x = self.activation(self.bn3c(self.conv3c(x)))
        x = self.ap3(x)

        x = x.view(-1, 128)
        if self.isL2:
            x = F.normalize(x)

        if self.double_output:
            return self.fc1(x), self.fc2(x), x
        else:
            return self.fc1(x), x


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    total_samples = clean_data.shape[0] * clean_data.shape[1]
    # 将数据转换为Tensor类型并移动到GPU上
    clean_data_tensor = torch.Tensor(clean_data.reshape(total_samples, -1)).to(device)
    clean_labels_tensor = torch.Tensor(clean_labels.reshape(total_samples, -1)).to(device)

    # 创建数据集和数据加载器
    dataset = TensorDataset(clean_data_tensor, clean_labels_tensor)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 实例化CNN模型并移动到GPU上
    model = CNN(num_classes=num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)  # 将数据移动到GPU上
            print(data.shape)
            optimizer.zero_grad()
            outputs, _ = model(data)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
