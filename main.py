import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

class CustomData(Dataset):
    def __init__(self, file_name):
        file_out = pd.read_csv(file_name)
        x = file_out.iloc[].values
        y = file_out.iloc[].values

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
    def __init__(self):
        super(MLP, self).__init__()
    def forward(self, x):
        pass

data = CustomData("")
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MLP().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

epochs = 10

model.train()
for epoch in range(epochs):
    losses = []
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

        if batch_num % 40 == 0:
            print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
    print('Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))