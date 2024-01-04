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

warnings.filterwarnings("ignore")

import numpy as np

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
    path_data = "Data/breast-cancer"
    name_json_info = "breast-cancer_info.json"
    name_json_rules = "breast-cancer_rules.json"
    name_model = "breast_cancer.pt"
    model_dropout = 0.1
    model_input_size = 30
    model_output_size = 2
    dataset = prepare_breast_cancer_dataset()

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
    y2E = blackbox.predict(X2E, device)
    y2E = np.asarray([dataset['possible_outcomes'][i] for i in y2E])
    for idx_record2explain in range(len(X2E)):
        print(idx_record2explain)
        explanation, infos = lore.explain(idx_record2explain, X2E, dataset, blackbox,
                                        ng_function=genetic_neighborhood,
                                        discrete_use_probabilities=True,
                                        continuous_function_estimation=False,
                                        returns_infos=True,
                                        path=path_data, sep=';', log=False)

        dfX2E = build_df2explain(blackbox, X2E, dataset).to_dict('records')
        dfx = dfX2E[idx_record2explain]

        #print('x = %s' % dfx)
        print('r = %s --> %s' % (explanation[0][1], explanation[0][0]))
        rule = {}
        rule["label"] = explanation[0][0][dataset['class_name']]
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
        rules_json.append(rule)

        #for delta in explanation[1]:
        #    print('delta', delta)

        #covered = lore.get_covered(explanation[0][1], dfX2E, dataset)
        #print(len(covered))
        #print(covered)

        #print(explanation[0][0][dataset['class_name']], '<<<<')

        def eval(x, y):
            return 1 if x == y else 0

        #precision = [1-eval(v, explanation[0][0][dataset['class_name']]) for v in y2E[covered]]
        #print(precision)
        #print(np.mean(precision), np.std(precision))
    with open(os.path.join("./json", name_json_rules), 'w') as f:
        json.dump(rules_json, f, cls=NpEncoder)

if __name__ == "__main__":
    main()
