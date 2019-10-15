import torch
import torch.nn as nn

class fc1(nn.Module):

    def __init__(self, num_classes=100):
        super(fc1, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(3*32*32, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
