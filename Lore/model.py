import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

class MLP(nn.Module):
    def __init__(self, dropout):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(8, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, 2),
        )
    
    def forward(self, x):
        return self.mlp(x)
    
    def predict(self, x, device):
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x.to(device))
        return np.argmax(pred.cpu().detach().numpy(), axis=1)
        