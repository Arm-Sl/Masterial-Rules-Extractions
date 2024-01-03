import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from Lore.model import MLP
from dataset import DiabetesDataset
import numpy as np

"""
MEILLEUR PARAM

learning_rate = 0.001
batch_size = 16
dropout 0
"""

"""param_grid = {
    "learning_rate": [0.1, 0.01, 0.001, 0.0001],
    "dropout": [0, 0.05, 0.1],
    "batch_size": [4, 8, 12, 16]
}"""
param_grid = {
    "learning_rate": [0.001],
    "dropout": [0],
    "batch_size": [16]
}
train_data = DiabetesDataset("Data/diabetes/diabetes.csv", "Data/diabetes/labels_diabetes.csv", split="Train")
valid_data = DiabetesDataset("Data/diabetes/diabetes.csv", "Data/diabetes/labels_diabetes.csv", split="Validation")
test_data = DiabetesDataset("Data/diabetes/diabetes.csv", "Data/diabetes/labels_diabetes.csv", split="Test")

param_list = list(ParameterGrid(param_grid))
min_loss = np.inf
best_dropout = 0
for params in param_list:
    learning_rate = params["learning_rate"]
    dropout = params["dropout"]
    batch_size = params["batch_size"]
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle = True)
    
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLP(8, 2, dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.000001)
    loss_fn = nn.CrossEntropyLoss()

    
    training_losses, valid_losses, accs = [],[],[]
    epochs = 25

    for epoch in range(epochs):
        model.train()
        training_loss = 0
        for i, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
                
            training_loss += loss.item()
        
        training_losses.append(training_loss)

        model.eval()
        valid_loss = 0
        acc = 0
        with torch.no_grad():
            for i, (data, labels) in enumerate(valid_loader):
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = loss_fn(outputs, labels)
                    
                valid_loss += loss.item()
                    
                _, predicted = outputs.topk(1, dim = 1)
                eq = predicted == labels.view(-1, 1)
                acc += eq.sum().item()
                    
            valid_losses.append(valid_loss)
            accs.append((acc/len(valid_data)) * 100)

        print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, valid acc : {:.2f}%'\
            .format(epoch+1, training_loss, valid_loss, (acc/len(valid_data)) * 100))
        
        if valid_loss <= min_loss:
            print("Saving Model {:.4f} ---> {:.4f}".format(min_loss, valid_loss))
            print(learning_rate, batch_size, dropout)
            best_dropout = dropout
            torch.save(model.state_dict(), "diabetes.pt")
            min_loss = valid_loss

test_loader = DataLoader(test_data, batch_size = 1, shuffle = True)
model = MLP(8, 2, best_dropout).to(device)
model.load_state_dict(torch.load("./diabetes.pt"))
total_correct = 0
with torch.no_grad():
    model.eval()
    for i, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)
        yhat = model(data)
        
        _, predicted = yhat.topk(1, dim = 1)
        eq = predicted == labels.view(-1, 1)
        total_correct += eq.sum().item()
        
        print("Predicted Value: {}..\tTrue Value: {}..".format(predicted.item(), labels.item()))

print("Score: {}/{}".format(total_correct, len(test_data)))
print("Percentage Correct: {:.2f}%".format((total_correct / len(test_data)) * 100))