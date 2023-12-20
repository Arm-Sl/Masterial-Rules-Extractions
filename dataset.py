import math
from torch.utils.data import Dataset
import torch
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

class CustomData(Dataset):
    def __init__(self, dataPath, labelPath):
        self.file_out_data = pd.read_csv(dataPath, sep=" ")
        self.file_out_label = pd.read_csv(labelPath, sep=" ")
        self.x = self.file_out_data.iloc[0:len(self.file_out_data.axes[0]), 0:len(self.file_out_data.axes[1])].values
        self.y = self.file_out_label.iloc[0:len(self.file_out_data.axes[0]), 0].values
        x_train = self.x
        y_train = self.y
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.Y_train = torch.tensor(y_train)

    def getDataset(self):
        categorical = self.file_out_data.select_dtypes(include=['object', 'category']).columns.tolist()
        continuous = self.file_out_data.select_dtypes(include='number').columns.tolist()
        distribution = {}
        for col in self.file_out_data.columns:
            distribution[col] = stats.ecdf(self.file_out_data[col]).cdf
        return {
            'X': self.file_out_data,
            'Y': self.file_out_label,
            'categorical': categorical,
            'continuous': continuous,
            'nbFeatures': len(self.file_out_data.columns),
            'nbCategorical': len(categorical),
            'distribution': distribution,
            'className': self.file_out_label.columns[0]
        }

    def getLine(self, index):
        return self.file_out_data.iloc[[index]]

    def __len__(self):
        return len(self.Y_train)
    
    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index]
    
    def getRepartition(self):
        train = math.floor(len(self) * 0.75)
        label = len(self) - train
        return [train, label]