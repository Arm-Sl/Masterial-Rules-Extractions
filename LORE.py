from model import MLP
from dataset import CustomData
import torch
import torch.nn as nn
from scipy.spatial.distance import sqeuclidean
import numpy as np
import pandas as pd
import random


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
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device

    def lore(self, x: pd.DataFrame, blackbox: MLP, N: int):
        G = 10
        pc = 0.5
        pm = 0.2
        Zsame, labelSame = self.geneticNeight(x, blackbox, self.fitnessSame, int(N/2), G, pc, pm)
        #Znot, labelNot = self.geneticNeight(x, blackbox, self.fitnessNotSame, int(N/2), G, pc, pm)
        #Z = Zsame + Znot
        #labels = labelSame + labelNot
        """for i in range(len(Z)):
            print(Z[i])
            print(labels[i])"""

        
    
    def buildTree(self, z):
        pass

    def extractRule(self, tree, x):
        pass

    def extractCounterFactual(self, tree, rules, x):
        pass

    def geneticNeight(self, x: pd.DataFrame, blackbox: MLP, fitness, N: int, G: int, pc: float, pm: float):
        # Initialisation de la population P0 avec N copie de x
        P = []
        for j in range(N):
            P.append({"data": x, "fitness": 0})
        i = 0
        self.evaluate(x, P, fitness, blackbox)
        while( i < G):
            P = self.select(P)
            P = self.crossover(P, pc)
            P = self.mutate(P, pm)
            self.evaluate(x, P, fitness, blackbox)
            i += 1
        Z = []
        label = []
        print("END")
        for p in P:
            Z.append(p["data"])
            label.append(blackbox.predict(p["data"], self.device))
        return Z, label
    
    def evaluate(self, x: pd.DataFrame, pop, fitness, blackbox: MLP):
        for p in pop:
            p["fitness"] = fitness(x, p["data"], blackbox)

    def select(self, pop) -> list:
        bestPop = []
        maxFitness = 0
        # Détermination du meilleur score de fitness
        for p in pop:
            if p["fitness"] > maxFitness:
                maxFitness = p["fitness"]
        # Sélection des individus
        for p in pop:
            if p["fitness"] >= 1:
                bestPop.append(p)
        return bestPop
 
    def crossover(self, pop: list, pc: float) -> list:
        num_parents = len(pop)
        num_crossovers = int(pc * num_parents)

        # Nombre d'individus sélectionnés pour le crossover est pair
        num_crossovers = num_crossovers if num_crossovers % 2 == 0 else num_crossovers - 1
        crossover_candidates_index = random.sample(range(0, num_parents), num_crossovers)

        # Crossover pour chaque paire d'individus sélectionnée
        for i in range(0, num_crossovers, 2):
            parent1 = pop[crossover_candidates_index[i]]
            parent2 = pop[crossover_candidates_index[i + 1]]
            child1, child2 = self.two_point_crossover(parent1["data"], parent2["data"])
            
            # Remplacez les parents par les enfants dans la population
            pop[crossover_candidates_index[i]]["data"] = child1
            pop[crossover_candidates_index[i + 1]]["data"] = child2

        return pop

    def two_point_crossover(self, parent1:pd.DataFrame, parent2: pd.DataFrame):
        if parent1.shape != parent2.shape:
            raise ValueError("Les parents doivent avoir le même nombre de colonnes")

        num_columns = parent1.shape[1]
        columns = parent1.columns
        # Choisissez deux points de crossover distincts
        point1 = random.randint(0, num_columns - 1)
        point2 = random.randint(0, num_columns - 1)
        while point1 == point2:
            point2 = random.randint(0, num_columns - 1)

        # Assurez-vous que point1 < point2
        if point1 > point2:
            point1, point2 = point2, point1

        # Effectuez le crossover
        c1 = parent1.iloc[:, point1:point2]
        c2 = parent2.iloc[:, point1:point2]
        child1 = pd.DataFrame([np.concatenate((parent1.iloc[:, :point1].values[0], c2.values[0], parent1.iloc[:, point2:].values[0]))], columns=columns)
        child2 = pd.DataFrame([np.concatenate((parent2.iloc[:, :point1].values[0], c1.values[0], parent2.iloc[:, point2:].values[0]))], columns=columns)
        return child1, child2

    def mutate(self, pop: list, pm: float) -> list:
        new_pop = []
        for p in pop:
            mutated = p.copy()
            for col in mutated["data"].columns:
                if random.random() <= pm:
                    mutated["data"].at[0, col] = self.mutate_gene(col)
            new_pop.append(mutated)
        return new_pop

    def mutate_gene(self, col):
        proba = random.random()
        i = 0
        while proba >= self.dataset["distribution"][col].probabilities[i]:
            i += 1
        return self.dataset["distribution"][col].quantiles[i]

    def fitnessSame(self, x: pd.DataFrame, z: pd.DataFrame, blackbox: MLP) -> float:
        return self.ISamepredict(x, z, blackbox) + (1 - self.distance(x, z)) - self.ISameData(x, z)

    def fitnessNotSame(self, x: pd.DataFrame, z: pd.DataFrame, blackbox: MLP) -> float:
        return self.INotSamepredict(x, z, blackbox) + (1 - self.distance(x, z)) - self.ISameData(x, z)
    
    def ISameData(self, x: pd.DataFrame, z: pd.DataFrame) -> int:
        # Attention les types des valeurs de x et z doivent être identique
        if x.equals(z):
            return 1
        return 0

    def ISamepredict(self, x: pd.DataFrame, z: pd.DataFrame, blackbox: MLP) -> int:
        xPredict = blackbox.predict(x, device)
        zPredict = blackbox.predict(z, device)
        if xPredict == zPredict:
            return 1
        return 0
    
    def INotSamepredict(self, x: pd.DataFrame, z: pd.DataFrame, blackbox: MLP) -> int:
        xPredict = blackbox.predict(x, device)
        zPredict = blackbox.predict(z, device)
        if xPredict != zPredict:
            return 1
        return 0

    def distance(self, x: pd.DataFrame, z: pd.DataFrame) -> float: 
        m = self.dataset["nbFeatures"]
        h = self.dataset["nbCategorical"]
        xCat = [x[att] for att in self.dataset["categorical"]]
        zCat = [z[att] for att in self.dataset["categorical"]]
        xCon = np.array(x[self.dataset["continuous"]])[0]
        zCon = np.array(z[self.dataset["continuous"]])[0]
        return (h/m)*self.simpleMatch(xCat, zCat) + ((m - h)/m)*self.normSquaredEuclid(xCon, zCon)
    
    def simpleMatch(self, x: pd.DataFrame, z: pd.DataFrame) -> float:
        if len(x) > 0 and len(z) > 0:
            count = 0
            for xi, zi in zip(x,z):
                if xi.values == zi.values:
                    count += 1
            return 1. - (count/len(x))
        return 0.
    
    def normSquaredEuclid(self, x: np.array, z: np.array) -> float:
        numerator = sqeuclidean((x-np.mean(x)),(z-np.mean(z)))
        denominator = (np.linalg.norm(x-np.mean(x)) ** 2) + (np.linalg.norm(z-np.mean(z)) ** 2)
        return 0.5*numerator/denominator 
    

data = CustomData("Data/diabetes/diabetes.csv", "Data/diabetes/labels_diabetes.csv")
x1 = data.getLine(0).reset_index(drop=True)
x2 = data.getLine(1).reset_index(drop=True)
x3 = data.getLine(2).reset_index(drop=True)
x4 = data.getLine(3).reset_index(drop=True)
pop =[{"data":x1, "fitness":1}, {"data":x2, "fitness":2},{"data":x3, "fitness":3}, {"data":x4, "fitness":4}]
lore = LORE(data.getDataset(), device)
lore.lore(x2, model, 10)