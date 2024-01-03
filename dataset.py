from torch.utils.data import Dataset
import torch
import pandas as pd

class DiabetesDataset(Dataset):
    def __init__(self, dataPath, labelPath, split="Train"):
        super(DiabetesDataset, self).__init__()
        self.df_x = pd.read_csv(dataPath, sep=" ")
        self.df_y = pd.read_csv(labelPath, sep=" ")

        if split == "Train":
            self.x = torch.from_numpy(self.df_x[:650].to_numpy()).type(torch.FloatTensor)
            self.y = torch.from_numpy(self.df_y[:650].to_numpy()).type(torch.LongTensor).squeeze(1)
            self.len = self.x.shape[0]
        elif split == "Validation":
            self.x = torch.from_numpy(self.df_x[650:710].to_numpy()).type(torch.FloatTensor)
            self.y = torch.from_numpy(self.df_y[650:710].to_numpy()).type(torch.LongTensor).squeeze(1)
            self.len = self.x.shape[0]
        elif split == "Test":
            self.x = torch.from_numpy(self.df_x[710:].to_numpy()).type(torch.FloatTensor)
            self.y = torch.from_numpy(self.df_y[710:].to_numpy()).type(torch.LongTensor).squeeze(1)
            self.len = self.x.shape[0]

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]