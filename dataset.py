import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd

class CustomData(Dataset):
    def __init__(self, dataPath, labelPath):
        file_out_data = pd.read_csv(dataPath, sep=" ")
        file_out_label = pd.read_csv(labelPath, sep=" ")
        x = file_out_data.iloc[0:len(file_out_data.axes[0]), 0:len(file_out_data.axes[1])].values
        y = file_out_label.iloc[0:len(file_out_data.axes[0]), 0].values
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y
        
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.Y_train = torch.tensor(y_train)
    
    def __len__(self):
        return len(self.Y_train)
    
    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index]
    
    def getRepartition(self):
        train = math.floor(len(self) * 0.75)
        label = len(self) - train
        return [train, label]