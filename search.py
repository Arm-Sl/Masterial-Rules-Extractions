import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from model import MLP
from dataset import CustomDataset
import numpy as np
import sys
"""
MEILLEUR PARAM DIABETES  78.71% sur test

learning_rate = 0.01
batch_size = 128
"""

"""
MEILLEUR PARAM BREAST CANCER 95.65% sur test

learning_rate = 0.01
batch_size = 64
"""

"""
MEILLEUR PARAM HEART 83.61% sur test

learning_rate = 0.01
batch_size = 64
"""

"""
MEILLEUR PARAM Covid 74.25% sur test

learning_rate = 0.001
batch_size = 512
dropout 0
"""

param_grid = {
    "dropout": [0, 0.1, 0.2],
    "learning_rate": [0.1, 0.01, 0.001],
    "batch_size": [64, 128, 256, 512]
}

if(len(sys.argv) < 2):
    print("Spécifier votre choix: breast, diabetes, heart, derm ou covid")
    exit()

match sys.argv[1]:
    case "breast":
        path_model = "./models/breast-cancer.pt"
        path_data = "Data/breast-cancer/breast-cancer.csv"
        path_labels = "Data/breast-cancer/labels_breast-cancer.csv"
        nb_features = 30
        nb_classe = 2
    case "diabetes":
        path_model = "./models/diabetes.pt"
        path_data = "Data/diabetes/diabetes.csv"
        path_labels = "Data/diabetes/labels_diabetes.csv"
        nb_features = 8
        nb_classe = 2
    case "heart":
        path_model = "./models/heart.pt"
        path_data = "Data/heart/heart.csv"
        path_labels = "Data/heart/labels_heart.csv"
        nb_features = 13
        nb_classe = 2
    case "derm":
        path_model = "./models/derm.pt"
        path_data = "Data/derm/derm.csv"
        path_labels = "Data/derm/labels_derm.csv"
        nb_features = 34
        nb_classe = 7
    case "covid":
        path_model = "./models/covid.pt"
        path_data = "Data/Covid-19/Covid-19.csv"
        path_labels = "Data/Covid-19/labels_Covid-19.csv"
        nb_features = 21
        nb_classe = 2
    case _:
        print("Mauvais arguments")
        exit()


train_data = CustomDataset(path_data, path_labels, split="Train")
valid_data = CustomDataset(path_data, path_labels, split="Validation")
test_data = CustomDataset(path_data, path_labels, split="Test")

param_list = list(ParameterGrid(param_grid))
min_loss = np.inf
best_dropout = 0
best_learning_rate = 0
best_batch_size = 0
for params in param_list:
    learning_rate = params["learning_rate"]
    dropout = params["dropout"]
    batch_size = params["batch_size"]
    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    valid_loader = DataLoader(valid_data, batch_size = batch_size, shuffle = True)
    
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLP(nb_features, nb_classe, dropout).to(device) #lancement modele

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
        
        if valid_loss < min_loss:
            print("Saving Model {:.4f} ---> {:.4f}".format(min_loss, valid_loss))
            best_learning_rate = learning_rate
            best_batch_size = batch_size
            best_dropout = dropout
            torch.save(model.state_dict(), path_model)
            min_loss = valid_loss

print("batch_size", best_batch_size)
print("learning_rate", best_learning_rate)
print("dropout", best_dropout)

test_loader = DataLoader(test_data, batch_size = 1, shuffle = True)
model = MLP(nb_features, nb_classe, best_dropout).to(device)
model.load_state_dict(torch.load(path_model))
total_correct = 0
with torch.no_grad():
    model.eval()
    for i, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)
        yhat = model(data)
        
        _, predicted = yhat.topk(1, dim = 1)
        eq = predicted == labels.view(-1, 1)
        total_correct += eq.sum().item()
        
        #print("Predicted Value: {}..\tTrue Value: {}..".format(predicted.item(), labels.item()))

print("Score: {}/{}".format(total_correct, len(test_data)))
print("Percentage Correct: {:.2f}%".format((total_correct / len(test_data)) * 100))