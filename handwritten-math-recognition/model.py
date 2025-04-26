import torch.nn as nn
import torch.nn.functional as F

class MathCNN(nn.Module):
    def __init__(self, num_classes=10):  
        super(MathCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 🟢 Yeni: Katmanları büyüttük!
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)               

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)               

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)               

        x = x.view(-1, 128 * 3 * 3)          
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
