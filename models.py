import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, fc_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, fc_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(fc_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x