import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from model import MLP
from dataset import CustomData

"""
MEILLEUR PARAM

learning_rate = 0.0005
batch_size = 12
dropout 0
"""


param_grid = {
    "learning_rate": [0.001, 0.01, 0.005, 0.0005],
    "dropout": [0, 0.1, 0.2, 0.3],
    "batch_size": [4, 8, 12, 16]
}
param_list = list(ParameterGrid(param_grid))
result = []
for params in param_list:
    learning_rate = params["learning_rate"]
    dropout = params["dropout"]
    batch_size = params["batch_size"]
    data = CustomData("Data/diabetes/diabetes.csv", "Data/diabetes/labels_diabetes.csv")
    training_data, test_data = torch.utils.data.random_split(data, data.getRepartition())
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MLP(dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    epochs = 15
    print(f"Lr: {learning_rate}, dropout: {dropout}, batchsize: {batch_size}")
    for epoch in range(epochs):
        losses = []
        model.train()
        for i, data in enumerate(train_dataloader, 0):
            x, y = data
            optimizer.zero_grad()

            x = x.to(device).float()
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        #print('Epoch %d | Loss %6.2f' % (epoch+1, sum(losses)/len(losses)))
    print(f'Accuracy of the network on the training data: {100 * (1 - (sum(losses)/len(losses)))} %')
    total = 0
    correct = 0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader,0):
            x, y = data
            x = x.to(device).float()
            y = y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    print(f'Accuracy of the network on the test data: {100 * correct / total} %')
    result.append({
        "model": model.state_dict(),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "dropout": dropout,
        "training_accuracy": 1 - (sum(losses)/len(losses)),
        "test_accuracy": correct / total
    })
max = 0
model = None
final_result = None
for r in result:
    if (r["test_accuracy"]+r["training_accuracy"])/2 > max:
        max = (r["test_accuracy"]+r["training_accuracy"])/2
        model = r["model"]
        final_result = r
print(final_result)
torch.save(model, "state_dict_model.pt")