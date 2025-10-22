# convnet.py
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, input_shape=(6, 60), num_classes=6):
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.backbone = nn.Sequential(
            nn.Conv1d(input_shape[0], 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(64, 64, 5),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        x = self.head(x)
        return x

if __name__ == "__main__":
    model = ConvNet()
    x = torch.randn(1, 6, 60)
    y = model(x)
    print("Output:", y.shape)
