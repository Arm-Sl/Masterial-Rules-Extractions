import torch.nn as nn
import numpy as np
import torch

class MLP(nn.Module):
    def __init__(self, input_size, nb_classes, dropout = 0):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(500, 750),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(750, 1000),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1000, 750),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(750, 500),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, nb_classes),
        )
    
    def forward(self, x):
        return self.mlp(x)
        
    
    def predict(self, x, device):
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x.to(device))
        return np.argmax(pred.cpu().detach().numpy(), axis=1)
        