import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, fc_size1, fc_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, fc_size1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_size1, fc_size2)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(fc_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x
    
class MinMSELoss(nn.Module):
    def __init__(self):
        super(MinMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')  # Compute MSE without reduction
    
    def forward(self, outputs, target_set):
        # Outputs shape: (1, output_size), Target set shape: (1, num_targets, output_size)
        mse_losses = self.mse_loss(outputs.unsqueeze(1), target_set)  # Shape: (1, num_targets, output_size)
        mse_losses = mse_losses.mean(dim=-1)  # Mean over output_size, resulting in shape: (1, num_targets)
        min_mse_loss = mse_losses.mediam(dim=1)[0]  # Minimum MSE over targets, shape: (1,)
        return min_mse_loss.mean() # Mean over batch

