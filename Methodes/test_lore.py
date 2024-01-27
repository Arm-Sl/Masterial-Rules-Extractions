import sys
import os
sys.path.append('../Masterial')
import lore
import torch
import model
from prepare_dataset import *
from neighbor_generator import *
import json
from sklearn.model_selection import train_test_split
from evaluation import evaluate_explanation

warnings.filterwarnings("ignore")

import numpy as np

"""
DIABETES

Completeness : 0.22580850522026993
Correctness : 0.9093386877164481
Fidelity : 0.7627323656735422
Robustesse : 0.9835752482811306
Number of rules : 1.0
Average rule length : 3.7941176470588234
"""

"""
BREAST

Completeness : 0.21052631578947364
Correctness : 0.965806446587152
Fidelity : 0.7541423001949318
Robustesse : 0.9563840155945419
Number of rules : 1.0
Average rule length : 2.537037037037037
"""

"""
HEART

Completeness : 0.18731523783929052
Correctness : 0.9532707046159312
Fidelity : 0.7519484009674817
Robustesse : 0.8656275194840097
Number of rules : 1.0
Average rule length : 3.9508196721311477
"""

"""
COVID
Completeness : 0.1599160583941606
Correctness : 0.9862611704521688
Fidelity : 0.7547901459854015
Robustesse : 0.729463503649635
Number of rules : 1.0
Average rule length : 3.4963503649635035
"""

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def main():
    if(len(sys.argv) < 2):
        print("Spécifier votre choix: breast, diabetes, heart ou covid")
        exit()

    match sys.argv[1]:
        case "breast":
            path_data = "Data/breast-cancer/"
            name_json_info = "breast-cancer_info.json"
            name_json_rules = "breast-cancer_lore_rules.json"
            name_model = "breast-cancer.pt"
            model_dropout = 0
            model_input_size = 30
            model_output_size = 2
            dataset = prepare_breast_cancer_dataset()
        case "diabetes":
            path_data = "Data/diabetes/"
            name_json_info = "diabetes_info.json"
            name_json_rules = "diabetes_lore_rules.json"
            name_model = "diabetes.pt"
            model_dropout = 0
            model_input_size = 8
            model_output_size = 2
            dataset = prepare_diabete_dataset()
        case "heart":
            path_data = "Data/heart/"
            name_json_info = "heart_info.json"
            name_json_rules = "heart_lore_rules.json"
            name_model = "heart.pt"
            model_dropout = 0
            model_input_size = 13
            model_output_size = 2
            dataset = prepare_heart_dataset()
        case "covid":
            path_data = "Data/Covid-19/"
            name_json_info = "Covid-19_info.json"
            name_json_rules = "Covid-19_lore_rules.json"
            name_model = "covid.pt"
            model_dropout = 0
            model_input_size = 21
            model_output_size = 2
            dataset = prepare_covid_dataset()
        case _:
            print("Mauvais arguments")
            exit()

    features = dataset["columns"][1:]
    info_json = {"class_values": dataset["possible_outcomes"],
                 "feature_names": features}
    rules_json = []
    with open(os.path.join("./json", name_json_info), 'w') as f:
        json.dump(info_json, f, cls=NpEncoder)

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    blackbox = model.MLP(model_input_size, model_output_size, model_dropout).to(device)
    blackbox.load_state_dict(torch.load(os.path.join("./models", name_model)))
    blackbox.eval()
    X2E = X_test
    y2E = blackbox.predict(X2E)
    nb = len(X2E)
    if sys.argv[1] == "covid":
        nb = 140

    completeness_l = []
    correctness_l = []
    fidelity_l = []
    robustesse_l = []
    number_of_rules_l = 0
    average_rule_length_l = []
    
    for idx_record2explain in range(nb):
        print(idx_record2explain)
        explanation, infos = lore.explain(idx_record2explain, X2E, dataset, blackbox,
                                        ng_function=genetic_neighborhood,
                                        discrete_use_probabilities=True,
                                        continuous_function_estimation=False,
                                        returns_infos=True,
                                        path=path_data, sep=';', log=False)

        dfX2E = build_df2explain(blackbox, X2E, dataset).to_dict('records')
        dfx = dfX2E[idx_record2explain]
        print('x = %s' % dfx)
        print('r = %s --> %s' % (explanation[0][1], explanation[0][0]))
        #infos["dt_dot"].write_png("tree.png")
        rule = {}
        for key, value in explanation[0][1].items():
            idx = info_json["feature_names"].index(key)
            supCount = value.count(">")
            infCount = value.count("<")
            if supCount == 1:
                # >173 ou >=173
                v = value.split(">")
                if "=" in v[1]:
                    v[1] = v[1][1:]
                rule[str(idx)] = [float(v[1]), np.inf]
            elif infCount == 1:
                # <14 ou <= 14
                v = value.split("<")
                if "=" in v[1]:
                    v[1] = v[1][1:]
                rule[str(idx)] = [-np.inf, float(v[1])]
            else:
                # 75< Blood < 80 ou 75<= Blood <= 80
                v = value.split("<")
                if "=" in v[2]:
                    v[2] = v[2][1:]
                rule[str(idx)] = [float(v[0]), float(v[2])]
        rule["label"] = explanation[0][0][dataset['class_name']]
        rules_json.append(rule)

        """for delta in explanation[1]:
            print('delta', delta)
        """
        tree_path = infos['tree_path']

        covered = lore.get_covered(explanation[0][1], dfX2E, dataset)


        def eval(x, y):
            return 1 if x == y else 0
        
        def count_zeros(arr):
            arr = np.array(arr)
            return arr.size - np.count_nonzero(arr)
        
        def perturb_data(data, delta=0.1):
            return data + np.random.uniform(low=-delta, high=delta, size=data.shape)

        # Correctness
        # Comparaison entre la classe original et la prédiction de la règle
        correct = [1-eval(dfX2E[idx][dataset["class_name"]], explanation[0][0][dataset['class_name']]) for idx in covered]

        # Fidelity
        # Prédictions de la règle sur l'ensemble de test
        f, _ = infos["predict"](dfX2E)
        # Comparaison de la prédiction à partir de la règle et de la prédiction du modèle sur l'ensemble de test
        fidel = [1-eval(y2E[idx], f[idx]) for idx in range(len(y2E))]
        
        # Robustesse
        # Perturbation des data
        X_perturbed = perturb_data(X2E)
        dfX2E_perturbed = build_df2explain(blackbox, X_perturbed, dataset).to_dict('records')
        # Prédiction de Lore sur les data perturbé
        r_perturbed, _ = infos["predict"](dfX2E_perturbed)
        # Comparaison entre prediction lore sur ensemble de test perturbé et les prédictions lore sur l'ensemble de test
        r = [1-eval(f[idx], r_perturbed[idx]) for idx in range(len(y2E))]

        if len(covered) > 0:
            completeness_l.append(len(covered) / len(X2E))
            correctness_l.append(count_zeros(correct) / len(covered))
            fidelity_l.append(count_zeros(fidel) / len(X2E))
            robustesse_l.append(count_zeros(r) / len(X2E))
            number_of_rules_l += 1
            average_rule_length_l.append(len(tree_path) - 1)

    print("Completeness :" ,np.mean(completeness_l))
    print("Correctness :" ,np.mean(correctness_l))
    print("Fidelity :" ,np.mean(fidelity_l))
    print("Robustesse :" ,np.mean(robustesse_l))
    print("Number of rules :" ,number_of_rules_l)
    print("Average rule length :" ,np.mean(average_rule_length_l))

    with open(os.path.join("./json", name_json_rules), 'w') as f:
        json.dump(rules_json, f, cls=NpEncoder)

if __name__ == "__main__":
    main()
