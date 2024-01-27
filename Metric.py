import json
import csv
import math

def Completeness(Dataset, Rulset):

    with open("json/" + Dataset + "_" + Rulset + "_rules.json") as mon_fichier:
        rules = json.load(mon_fichier)

    first = True
    covered = 0
    total = 0
    labels = []

    with open("Data/" + Dataset + "/" + Dataset + ".csv", newline='') as csvfile:
        loadData = csv.reader(csvfile, delimiter=' ', quotechar='|')

        for row in loadData:
            if not first:  # on ne prens pas la première ligne du csv
                total += 1
                for r in rules:
                    valide = True
                    for i in range(len(r) - 1):  # la dernière valeur est le label

                        j = list(r.items())[i][0]  # numérot de la colone concerné

                        if r[j][1] == math.inf:
                            if float(row[int(j)]) < r[j][0]:
                                valide = False
                                break
                        else:
                            if float(row[int(j)]) > r[j][1]:
                                valide = False
                                break
                    if valide:
                        if r not in labels:
                            labels.append(r)
                        break

                if valide:
                    covered += 1

            else:
                first = False

    #print(covered, " instance couverte sur ", total, " donc Completeness = ", covered / total)
    return covered / total

def Correctness(Dataset, Rulset):

    with open("json/" + Dataset + "_" + Rulset + "_rules.json") as mon_fichier:
        rules = json.load(mon_fichier)

    first = True
    covered = 0
    total = 0
    label = dict()

    labels = []

    with open("Data/" + Dataset + "/" + Dataset + ".csv", newline='') as csvfile:
        loadData = csv.reader(csvfile, delimiter=' ', quotechar='|')

        for row in loadData:
            if not first:  # on ne prens pas la première ligne du csv
                total += 1

                for r in rules:
                    valide = True
                    for i in range(len(r) - 1):  # la dernière valeur est le label

                        j = list(r.items())[i][0]  # numérot de la colone concerné

                        if r[j][1] == math.inf:
                            if float(row[int(j)]) < r[j][0]:
                                valide = False
                                break
                        else:
                            if float(row[int(j)]) > r[j][1]:
                                valide = False
                                break
                    if valide:
                        covered += 1
                        label[total+1] = r["label"]
                        if r not in labels:
                            labels.append(r)
                        break

            else:
                first = False

    i = 0
    correct = 0
    first = True

    with open("Data/" + Dataset + "/labels_" + Dataset + ".csv", newline='') as csvfile:
        loadData = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in loadData:

            if not first:
                if i in label:
                    if int(row[0]) == label[i]:
                        correct += 1
                i += 1
            else:
                first = False

    #print(correct, " instance corectement classifier sur ", total, " donc Completeness = ", correct / total)
    return correct / total

def Fidelity(Dataset, Rulset):
    with open("json/" + Dataset + "_" + Rulset + "_rules.json") as mon_fichier:
        rules = json.load(mon_fichier)

    first = True
    covered = 0
    total = 0
    label = dict()

    labels = []

    with open("Data/" + Dataset + "/" + Dataset + ".csv", newline='') as csvfile:
        loadData = csv.reader(csvfile, delimiter=' ', quotechar='|')

        for row in loadData:
            if not first:  # on ne prens pas la première ligne du csv
                total += 1

                for r in rules:
                    valide = True
                    for i in range(len(r) - 1):  # la dernière valeur est le label

                        j = list(r.items())[i][0]  # numérot de la colone concerné

                        if r[j][1] == math.inf:
                            if float(row[int(j)]) < r[j][0]:
                                valide = False
                                break
                        else:
                            if float(row[int(j)]) > r[j][1]:
                                valide = False
                                break
                    if valide:
                        covered += 1
                        label[total + 1] = r["label"]
                        if r not in labels:
                            labels.append(r)
                        break

            else:
                first = False

    i = 0
    correct = 0
    first = True

    with open("Data/" + Dataset + "/" + Dataset + "-predictions.csv", newline='') as csvfile: #remplacer par les prédictions du model
        loadData = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in loadData:
            if not first:
                if i in label:
                    if int(row[0]) == label[i]:
                        correct += 1
                i += 1
            else:
                first = False

    #print(correct, " instance corectement classifier sur ", total, " donc Completeness = ", correct / total)
    return correct / total

def Robustness(Dataset, Rulset, delta):
    with open("json/" + Dataset + "_" + Rulset + "_rules.json") as mon_fichier:
        rules = json.load(mon_fichier)

    first = True
    total = 0
    label = dict()

    with open("Data/" + Dataset + "/" + Dataset + ".csv", newline='') as csvfile:
        loadData = csv.reader(csvfile, delimiter=' ', quotechar='|')

        for row in loadData:
            if not first:  # on ne prens pas la première ligne du csv
                total += 1

                for r in rules:
                    valide = True
                    for i in range(len(r) - 1):  # la dernière valeur est le label

                        j = list(r.items())[i][0]  # numérot de la colone concerné

                        if r[j][1] == math.inf:
                            if float(row[int(j)]) < r[j][0]:
                                valide = False
                                break
                        else:
                            if float(row[int(j)]) > r[j][1]:
                                valide = False
                                break
                    if valide:
                        label[total+1] = r["label"]
                        break

            else:
                first = False

    first = True
    total = 0
    labelPerturbed = dict()

    # on reffeer la même chose mai en modifiant data
    with open("Data/" + Dataset + "/" + Dataset + ".csv", newline='') as csvfile:
        loadData = csv.reader(csvfile, delimiter=' ', quotechar='|')

        for row in loadData:
            if not first:  # on ne prens pas la première ligne du csv
                total += 1

                for r in rules:
                    valide = True
                    for i in range(len(r) - 1):  # la dernière valeur est le label

                        j = list(r.items())[i][0]  # numérot de la colone concerné

                        if r[j][1] == math.inf:
                            if (float(row[int(j)]) + float(row[int(j)]) * delta) < r[j][0]:
                                valide = False
                                break
                        else:
                            if (float(row[int(j)]) + float(row[int(j)]) * delta) > r[j][1]:
                                valide = False
                                break
                    if valide:
                        labelPerturbed[total + 1] = r["label"]
                        break

            else:
                first = False

    #compare label et labelPerturbed

    same = 0
    for i in range(total):
        if i in label and i in labelPerturbed:
            if label[i] == labelPerturbed[i]: same += 1

    #print(same, " régle non changer sur ", total, " donc Robustness = ", same / total)
    return same / total

def NumberOfRules(Dataset, Rulset):
    with open("json/" + Dataset + "_" + Rulset + "_rules.json") as mon_fichier:
        rules = json.load(mon_fichier)
    #print(len(rules), " règle ont été écrite")
    return len(rules)

def AverageRuleLength(Dataset, Rulset):
    with open("json/" + Dataset + "_" + Rulset + "_rules.json") as mon_fichier:
        rules = json.load(mon_fichier)

    total = 0
    taille = 0

    for r in rules:
        total += 1
        taille += len(r)-1  # la taille du dictionnaire - le label

    #print("la longeur moyenne des règles est de ", taille/total)
    return taille/total

def FractionOfClasses(Dataset, Rulset):
    with open("Data/" + Dataset + "/featureNames_" + Dataset + ".csv", newline='') as csvfile:
        loadData = csv.reader(csvfile, delimiter=' ', quotechar='|')

        nbClasses = 0

        for row in loadData: nbClasses += 1

    with open("json/" + Dataset + "_" + Rulset + "_rules.json") as mon_fichier:
        rules = json.load(mon_fichier)

    CoveredClasses = []

    for r in rules:
        for i in range(nbClasses):
            if str(i) in r and i not in CoveredClasses: CoveredClasses.append(i)

    #print(len(CoveredClasses), " classe  sont couverte sur ", nbClasses, " donc FractionOfClasses = ", len(CoveredClasses)/nbClasses)*
    return len(CoveredClasses)/nbClasses

def FractionOverlap(Dataset, Rulset):
    with open("json/" + Dataset + "_" + Rulset + "_rules.json") as mon_fichier:
        rules = json.load(mon_fichier)

    res = [len(rules)]

    with open("Data/" + Dataset + "/" + Dataset + ".csv", newline='') as csvfile:
        loadData = csv.reader(csvfile, delimiter=' ', quotechar='|')
        total = 0
        overlap = 0
        first = True
        for row in loadData:
            if not first:
                total += 1

                for i in range(len(rules)):
                    for j in range(i + 1, len(rules)):
                        valide1 = True
                        for r in rules[i]:

                            if r != "label":
                                if rules[i][r][1] == math.inf:
                                    if float(row[int(r)]) < rules[i][r][0]:
                                        valide1 = False
                                        break
                                else:
                                    if float(row[int(r)]) > rules[i][r][1]:
                                        valide1 = False
                                        break

                        valide2 = True
                        for r in rules[j]:
                            if r != "label":
                                if rules[j][r][1] == math.inf:
                                    if float(row[int(r)]) < rules[j][r][0]:
                                        valide2 = False
                                        break
                                else:
                                    if float(row[int(r)]) > rules[j][r][1]:
                                        valide2 = False
                                        break

                        if valide1 and valide2: overlap += 1

            else: first = False
    #print("FractionOverlap = ", 2 / ((len(rules)) * (len(rules) - 1)) * (overlap / total))
    if overlap == 0 : return 0
    return 2 / ((len(rules)) * (len(rules) - 1)) * (overlap / total)


if __name__ == '__main__':
    dataset = "breast-cancer"
    rulset = "lore"

    print(dataset, rulset)
    print("Completeness = ", Completeness(dataset, rulset))
    print("Fidelity = ", Fidelity(dataset, rulset))
    print("Correctness = ", Correctness(dataset, rulset))
    print("Robustness = ", Robustness(dataset, rulset, 0.01))
    print("NumberOfRules = ", NumberOfRules(dataset, rulset))
    print("AverageRuleLength = ", AverageRuleLength(dataset, rulset))
    print("FractionOfClasses = ", FractionOfClasses(dataset, rulset))
    print("FractionOverlap = ", FractionOverlap(dataset, rulset))

