import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNet(nn.Module):
    def __init__(self, params):
        super().__init__()
        if params.data_num_dimensions < 13:
            self.fc = nn.Linear(params.data_num_dimensions, 128)
            self.fc1 = nn.Linear(128, 64)
            self.fc2 = nn.Linear(64, params.num_classes)
            self.dropout = nn.Dropout(p=0.2)
        else:
            self.fc = nn.Linear(params.data_num_dimensions, 4 * params.data_num_dimensions)
            self.fc1 = nn.Linear(4 * params.data_num_dimensions, 2 * params.data_num_dimensions)
            self.fc2 = nn.Linear(2 * params.data_num_dimensions, params.num_classes)
            self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
