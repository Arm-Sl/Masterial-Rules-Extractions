from model import MLP
from dataset import CustomData
import torch
from scipy.spatial.distance import sqeuclidean
import numpy as np
import pandas as pd
import random
import copy
import yadt
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(0).to(device)
model.load_state_dict(torch.load("./state_dict_model.pt", map_location=torch.device('cpu')))
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
        Znot, labelNot = self.geneticNeight(x, blackbox, self.fitnessNotSame, int(N/2), G, pc, pm)
        Z = Zsame + Znot
        labels = labelSame + labelNot
        self.buildTree(Z, labels)
        
    
    def buildTree(self, z, labels):
        predict = []
        for l in labels:
            if l == [0]:
                predict.append(pd.DataFrame(["No_Diabete"],columns=[self.dataset["className"]]))
            elif l == [1]:
                predict.append(pd.DataFrame(["Diabete"], columns=[self.dataset["className"]]))
        y = pd.concat(predict).reset_index(drop=True)
        x = pd.concat(z).reset_index(drop=True)
        df = pd.concat([x,y], axis=1)
        

        df1 = pd.concat([self.dataset["X"],self.dataset["Y"]], axis=1)
        md = yadt.meta_data(df)
        #yadt.to_yadt(x.values, md, y=y.squeeze(), filenames='np_dataset.names', filedata='np_dataset.data.gz')
        clf_yadt = yadt.YaDTClassifier(md, options='-m 8 -grpure')
        #clf_yadt.fit(self.dataset["X"], self.dataset["Y"].squeeze())
        clf_yadt.fit(x, y.squeeze())
        pickle.dump(clf_yadt, open("clf_yadt2.p", "wb"))

        new_cfl = pickle.load(open("clf_yadt2.p", "rb"))
        print(new_cfl.get_tree())

    def extractRule(self, tree, x):
        pass

    def extractCounterFactual(self, tree, rules, x):
        pass

    def geneticNeight(self, x: pd.DataFrame, blackbox: MLP, fitness, N: int, G: int, pc: float, pm: float):
        # Initialisation de la population P0 avec N copie de x
        P = []
        for j in range(N):
            P.append({"data": copy.deepcopy(x), "fitness": 0})
        i = 0
        P = self.evaluate(x, P, fitness, blackbox)
        while( i < G):
            P = self.select(P)
            P = self.crossover(P, pc)
            P = self.mutate(P, pm)
            P = self.evaluate(x, P, fitness, blackbox)
            i += 1
        P = self.select(P)
        Z = []
        label = []
        for p in P:
            Z.append(p["data"])
            label.append(blackbox.predict(p["data"], self.device))
        return Z, label
    
    def evaluate(self, x: pd.DataFrame, pop, fitness, blackbox: MLP):
        new_pop = []
        for p in pop:
            new_ind = p.copy()
            new_ind["fitness"] = fitness(x, new_ind["data"], blackbox)
            new_pop.append(new_ind)
        return new_pop

    def select(self, pop) -> list:
        bestPop = []
        maxFitness = 0
        # Détermination du meilleur score de fitness
        for p in pop:
            if round(p["fitness"],2) > maxFitness:
                maxFitness = round(p["fitness"],2)
        # Sélection des individus
        for p in pop:
            if round(p["fitness"], 2) == maxFitness:
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
            if random.random() < pm:
                cols = mutated["data"].columns
                r = random.randint(0, len(cols)-1)
                mutated["data"].at[0, cols[r]] = self.mutate_gene(cols[r])
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
x1 = data.getLine(0)
x2 = data.getLine(1)
x3 = data.getLine(2)
lore = LORE(data.getDataset(), device)
lore.lore(x1, model, 2000)