from torch.utils.data import Dataset
import torch
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, dataPath, labelPath, split="Train"):
        super(CustomDataset, self).__init__()
        self.df_x = pd.read_csv(dataPath, sep=" ").sample(frac=1)
        self.df_y = pd.read_csv(labelPath, sep=" ").sample(frac=1)
        nbTrain = int(self.df_x.shape[0] * 0.75)
        nbVal = nbTrain + int(self.df_x.shape[0] * 0.15)
        if split == "Train":
            self.x = torch.from_numpy(self.df_x[:nbTrain].to_numpy()).type(torch.FloatTensor)
            self.y = torch.from_numpy(self.df_y[:nbTrain].to_numpy()).type(torch.LongTensor).squeeze(1)
            self.len = self.x.shape[0]
        elif split == "Validation":
            self.x = torch.from_numpy(self.df_x[nbTrain:nbVal].to_numpy()).type(torch.FloatTensor)
            self.y = torch.from_numpy(self.df_y[:nbTrain].to_numpy()).type(torch.LongTensor).squeeze(1)
            self.len = self.x.shape[0]
        elif split == "Test":
            self.x = torch.from_numpy(self.df_x[nbVal:].to_numpy()).type(torch.FloatTensor)
            self.y = torch.from_numpy(self.df_y[:nbTrain].to_numpy()).type(torch.LongTensor).squeeze(1)
            self.len = self.x.shape[0]

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]