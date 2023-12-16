from model import MLP
from dataset import CustomData
import torch
import torch.nn as nn
from scipy.spatial.distance import sqeuclidean
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(0).to(device)
model.load_state_dict(torch.load("./state_dict_model.pt"))
model.eval()
#print(data.__getitem__(0))
#predict = model(data.__getitem__(0)[0])
#print(torch.argmax(predict.data))
"""
Format a renvoyer tuple -> ({"feature1": value, "feature2": value}, y)
"""

class LORE():
    def __init__(self, dataset):
        self.dataset = dataset

    def lore(self, x, b):
        pass
    
    def buildTree(self, z):
        pass

    def extractRule(self, tree, x):
        pass

    def extractCounterFactual(self, tree, rules, x):
        pass

    def geneticNeight(self, x, fitness, b, N, G, pc, pm):
        pass
    
    def evaluate(self, pop, fitness, b):
        pass

    def select(self, pop):
        pass

    def crossover(self, pop, pc):
        pass

    def mutate(self, pop, pm):
        pass

    def fitness(self, x, b, z):
        return self.I(torch.argmax(b(x).data), torch.argmax(b(z).data)) + (1 - self.distance(x, z)) - self.I(x, z)
    
    def I(self, a, b, diff=False):
        if diff:
            if not torch.equal(a, b):
                return 1
            return 0
        else:
            if torch.equal(a, b):
                return 1
            return 0

    def distance(self, x, z):
        m = self.dataset["nbFeatures"]
        h = self.dataset["nbCategorical"]
        xCat = [x[att] for att in self.dataset["categorical"]]
        zCat = [z[att] for att in self.dataset["categorical"]]
        xCon = np.array(x[d["continuous"]])[0]
        zCon = np.array(z[d["continuous"]])[0]
        return (h/m)*self.simpleMatch(xCat, zCat) + ((m - h)/m)*self.normSquaredEuclid(xCon, zCon)
    
    def simpleMatch(self, x, z) -> float:
        if len(x) > 0 and len(z) > 0:
            count = 0
            for xi, zi in zip(x,z):
                if xi.values == zi.values:
                    count += 1
            return 1 - (count/len(x))
        return 0.
    
    def normSquaredEuclid(self, x: np.array, z: np.array) -> float:
        numerator = sqeuclidean((x-np.mean(x)),(z-np.mean(z)))
        denominator = (np.linalg.norm(x-np.mean(x)) ** 2) + (np.linalg.norm(z-np.mean(z)) ** 2)
        return 0.5*numerator/denominator 
    

data = CustomData("Data/diabetes/diabetes.csv", "Data/diabetes/labels_diabetes.csv")
print(data.x_train)
x1 = data.getLine(0)
x2 = data.getLine(1)
print(model.predict(x1))
lore = LORE(data.getDataset())
