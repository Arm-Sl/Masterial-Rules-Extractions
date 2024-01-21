import torch.nn as nn
import numpy as np
import torch

class MLP(nn.Module):
    def __init__(self, input_size, nb_classes, dropout = 0):
        super(MLP, self).__init__()
        self.f_connected1 = nn.Linear(input_size, 16)
        self.f_connected2 = nn.Linear(16, 32)
        self.f_connected3 = nn.Linear(32, 64)
        self.f_connected6 = nn.Linear(64, 32)
        self.f_connected7 = nn.Linear(32, 16)
        self.out = nn.Linear(16, nb_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.f_connected1(x))
        x = self.relu(self.f_connected2(x))
        x = self.relu(self.f_connected3(x))
        x = self.relu(self.f_connected6(x))
        x = self.relu(self.f_connected7(x))
        x = self.out(x)
        return x
        
    
    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pred = self.forward(x.to(device))
        return np.argmax(pred.cpu().detach().numpy(), axis=1)
        