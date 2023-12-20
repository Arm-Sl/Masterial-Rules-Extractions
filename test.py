import random
import pandas as pd
import numpy as np

def two_point_crossover(parent1, parent2):
    if parent1.shape[0] != parent2.shape[0]:
        raise ValueError("Les parents doivent avoir le même nombre de colonnes.")

    point1 = random.randint(0, parent1.shape[1] - 1)
    point2 = random.randint(0, parent1.shape[1] - 1)
    while point1 == point2:
        point2 = random.randint(0, parent1.shape[1] - 1)

    if point1 > point2:
        point1, point2 = point2, point1

    child1 = pd.concat([parent1.iloc[:, :point1], parent2.iloc[:, point1:point2], parent1.iloc[:, point2:]], axis=1)
    child2 = pd.concat([parent2.iloc[:, :point1], parent1.iloc[:, point1:point2], parent2.iloc[:, point2:]], axis=1)

    return child1, child2

def apply_crossover(population, pc):
    num_parents = len(population)
    num_crossovers = int(pc * num_parents)

    # Assurez-vous que le nombre d'individus sélectionnés pour le crossover est pair
    num_crossovers = num_crossovers if num_crossovers % 2 == 0 else num_crossovers - 1

    # Sélectionnez aléatoirement les individus pour le crossover
    crossover_candidates = random.sample(population, num_crossovers)

    # Effectuez le crossover pour chaque paire d'individus sélectionnée
    for i in range(0, num_crossovers, 2):
        parent1 = crossover_candidates[i]
        parent2 = crossover_candidates[i + 1]
        child1, child2 = two_point_crossover(parent1, parent2)

        # Remplacez les parents par les enfants dans la population
        population[population.index(parent1)] = child1
        population[population.index(parent2)] = child2

    return population

# Exemple d'utilisation avec des DataFrames
columns = ['gene1', 'gene2', 'gene3', 'gene4', 'gene5']
population_size = 6
"""population = [pd.DataFrame({'gene1': random.randint(1, 6),
                             'gene2': random.randint(6, 11),
                             'gene3': random.randint(11, 16),
                             'gene4': random.randint(16, 21),
                             'gene5': random.randint(21, 26)}) for _ in range(population_size)]"""
population = [pd.DataFrame(np.random.randint(0, 100, size=(1, 5)), columns=columns) for _ in range(population_size)]
pc = 0.8  # Probabilité de crossover

print("Population avant crossover:")
for ind in population:
    print(ind)

# Appliquez le crossover avec une probabilité pc
population = apply_crossover(population, pc)

print("\nPopulation après crossover:")
for ind in population:
    print(ind)
