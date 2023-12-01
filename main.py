import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid

class DiabeteData(Dataset):
    def __init__(self):
        file_out_data = pd.read_csv("Data/diabetes/diabetes.csv", sep=" ")
        file_out_label = pd.read_csv("Data/diabetes/labels_diabetes.csv", sep=" ")
        x = file_out_data.iloc[0:768, 0:8].values
        y = file_out_label.iloc[0:768, 0].values
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y
        
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.Y_train = torch.tensor(y_train)
    
    def __len__(self):
        return len(self.Y_train)
    
    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index]
    
class HeartData(Dataset):
    def __init__(self):
        file_out_data = pd.read_csv("Data/heart/heart.csv", sep=" ")
        file_out_label = pd.read_csv("Data/heart/labels_heart.csv", sep=" ")
        x = file_out_data.iloc[1:303, 1:13].values
        y = file_out_label.iloc[1:303, 1].values
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y
        
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.Y_train = torch.tensor(y_train)
    
    def __len__(self):
        return len(self.Y_train)
    
    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index]
    
class BreastCancerData(Dataset):
    def __init__(self):
        file_out_data = pd.read_csv("Data/breast-cancer/breast-cancer.csv", sep=" ")
        file_out_label = pd.read_csv("Data/breast-cancer/labels_breast-cancer.csv", sep=" ")
        x = file_out_data.iloc[1:569, 1:30].values
        y = file_out_label.iloc[1:569, 1].values
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y
        
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.Y_train = torch.tensor(y_train)
    
    def __len__(self):
        return len(self.Y_train)
    
    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index]

class CovidData(Dataset):
    def __init__(self):
        file_out_data = pd.read_csv("Data/Covid-19/Covid-19.csv", sep=" ")
        file_out_label = pd.read_csv("Data/Covid-19/labels_Covid-19.csv", sep=" ")
        x = file_out_data.iloc[1:20000, 1:22].values
        y = file_out_label.iloc[1:20000, 1].values
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y
        
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.Y_train = torch.tensor(y_train)
    
    def __len__(self):
        return len(self.Y_train)
    
    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index]

class DermData(Dataset):
    def __init__(self):
        file_out_data = pd.read_csv("Data/derm/derm.csv", sep= " ")
        file_out_label = pd.read_csv("Data/derm/labels_derm.csv", sep= " ")
        x = file_out_data.iloc[1:366,1:8].values
        y = file_out_label.iloc[1:366,1].values

        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = y
        
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.Y_train = torch.tensor(y_train)
    
    def __len__(self):
        return len(self.Y_train)
    
    def __getitem__(self, index):
        return self.X_train[index], self.Y_train[index]

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

"""
MEILLEUR PARAM

learning_rate = 0.001
batch_size = 4
dropout 0.1
"""


param_grid = {
    "learning_rate": [0.001],
    "dropout": [0.1],
    "batch_size": [4]
}
param_list = list(ParameterGrid(param_grid))
result = []
for params in param_list:
    learning_rate = params["learning_rate"]
    dropout = params["dropout"]
    batch_size = params["batch_size"]
    data = DiabeteData()

    training_data, test_data = torch.utils.data.random_split(data, [576, 192])
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLP(dropout).to(device) #model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #optimiseur
    criterion = nn.CrossEntropyLoss()  #fonction de cout

    epochs = 50
    
    for epoch in range(epochs):
        losses = []
        model.train()
        for batch_num, input_data in enumerate(train_dataloader):
            optimizer.zero_grad()
            x, y = input_data
            x = x.to(device).float()
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        print('Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))
    result.append({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "dropout": dropout,
        "loss": sum(losses)/len(losses)
        })
    print({
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "dropout": dropout,
        "loss": sum(losses)/len(losses)
        })
print(result)
