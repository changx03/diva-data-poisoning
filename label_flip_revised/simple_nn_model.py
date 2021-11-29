import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    """A simple fullly-connected neural network with 1 hidden-layer"""

    def __init__(self, input_dim, hidden_dim=128, output_dim=2):
        super(SimpleModel, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x
