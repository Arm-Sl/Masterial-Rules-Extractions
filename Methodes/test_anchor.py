import sys
import os
sys.path.append('../Masterial')
import torch
torch.cuda.empty_cache()
import model
from prepare_dataset import *
from neighbor_generator import *
import json
from anchor import anchor_tabular
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
    if(len(sys.argv) < 2):
        print("Spécifier votre choix: breast, diabetes, heart ou covid")
        exit()
    
    
    match sys.argv[1]:
        case "breast":
            path_data = "Data/breast-cancer/"
            name_json_info = "breast-cancer_info.json"
            name_json_rules = "breast-cancer_anchor_rules.json"
            name_model = "breast-cancer.pt"
            model_dropout = 0.1
            model_input_size = 30
            model_output_size = 2
            dataset = prepare_breast_cancer_dataset()
        case "diabetes":
            path_data = "Data/diabetes/"
            name_json_info = "diabetes_info.json"
            name_json_rules = "diabetes_anchor_rules.json"
            name_model = "diabetes.pt"
            model_dropout = 0
            model_input_size = 8
            model_output_size = 2
            dataset = prepare_diabete_dataset()
        case "heart":
            path_data = "Data/heart/"
            name_json_info = "heart_info.json"
            name_json_rules = "heart_anchor_rules.json"
            name_model = "heart.pt"
            model_dropout = 0.01
            model_input_size = 13
            model_output_size = 2
            dataset = prepare_heart_dataset()
        case "covid":
            path_data = "Data/Covid-19/"
            name_json_info = "Covid-19_info.json"
            name_json_rules = "Covid-19_anchor_rules.json"
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
    listf = list()
    nb = len(X2E)
    if sys.argv[1] == "covid":
        nb = 140 #on explique les 140 premiers exemples du dataset test (pour des raisons de temps de calcul) uniquement COVID
   
    completeness_a = []
    correctness_a = []
    fidelity_a = []
    robustness_a = []
    number_of_rules_a = []
    average_rule_length_a = []
   
    for idx_record2explain in range(nb):

        class_name = dataset['class_name']
        columns = dataset['columns']
        continuous = dataset['continuous']
        possible_outcomes = dataset['possible_outcomes']
        label_encoder = dataset['label_encoder']

        dfX2E = build_df2explain(blackbox, X2E, dataset).to_dict('records') #aucune idéeeeee 


        feature_names = list(columns)
        feature_names.remove(class_name)

        categorical_names = dict()
        idx_discrete_features = list()
        for idx, col in enumerate(feature_names):
            if col == class_name or col in continuous:
                continue
            idx_discrete_features.append(idx)
            categorical_names[idx] = label_encoder[col].classes_

        # Create Anchor Explainer
        explainer = anchor_tabular.AnchorTabularExplainer(possible_outcomes, feature_names, X2E, categorical_names)
        torch.cuda.empty_cache()

        explainer.fit(X_train, y_train, X_test, y_test)

        pred = possible_outcomes[blackbox.predict(X2E[idx_record2explain].reshape(1, -1))[0]]
        #print('Prediction: ', pred)

        exp, info = explainer.explain_instance(X2E[idx_record2explain].reshape(1, -1), blackbox.predict, threshold=0.95)

        #print('Anchor: %s' % (' AND '.join(exp.names())))
        #print(exp.names())

        text = exp.names()
        res = list()
        for txt in text:   
            l = list()
            x = txt.split(sep = " ")
            for t in x:
                if t.replace(".","",1).isdigit():
                    t = float(t)
                l.append(t)
            res.append(l)
        #print(res)

        d=dict()
        for r in res:     #a partir de la liste de chaine de caractere res on transforme en dictionnaire de la forme {feature:[min,max]}
            if type(r[0]) == float:
                idx = info_json["feature_names"].index(r[2]) #on recupere l'indice de la feature dans le dataset
                d[idx]=[r[0],r[4]]
            else :
                idx = info_json["feature_names"].index(r[0])
                if r[1]==">"or r[1]==">=":
                    d[idx]=[r[2],np.inf]
                else:
                    d[idx]=[-np.inf,r[2]]
        d["label"]=pred
        listf.append(d)
        #print(listf)


        #print('Precision: %.2f' % exp.precision())
        #print('Coverage: %.2f' % exp.coverage())

        def eval(x, y):
            return 1 if x == y else 0
        
        def count_zeros(arr):
            arr = np.array(arr)
            return arr.size - np.count_nonzero(arr)
        
        def perturb_data(data, delta=0.1):
            return data + np.random.uniform(low=-delta, high=delta, size=data.shape)


        # Get test examples where the anchor applies
        fit_anchor = np.where(np.all(X2E[:, exp.features()] == X2E[idx_record2explain][exp.features()], axis=1))[0]
        completeness_a.append(fit_anchor.shape[0] / float(X2E.shape[0]))

        precision = np.mean(blackbox.predict(X2E[fit_anchor]) ==  blackbox.predict(X2E[idx_record2explain].reshape(1, -1)))
        correctness_a.append(precision)
        
        # Fidelity
        # Prédictions de la règle sur l'ensemble de test
        #f, _ = info["predict"](dfX2E)
        # Comparaison de la prédiction à partir de la règle et de la prédiction du modèle sur l'ensemble de test
        #fidel = [1-eval(y2E[idx], f[idx]) for idx in range(len(y2E))]
        #fidelity_a.append(count_zeros(fidel) / len(X2E))


        # Robustness
        perturbed_data = perturb_data(X2E[idx_record2explain].reshape(1, -1))
        fit_anchor_perturbed = np.where(np.all(perturbed_data[:, exp.features()] == X2E[idx_record2explain][exp.features()], axis=1))[0]
        robustness = fit_anchor_perturbed.shape[0] / float(perturbed_data.shape[0])
        robustness_a.append(robustness)



        raw_data = info['state']['raw_data']
        batch_size = len(raw_data) // 4
        

        for i in range(0, len(raw_data), batch_size):
            batch = raw_data[i:i+batch_size]
            predictions = blackbox.predict(batch)
            #print(predictions)

    print("Completeness :" ,np.mean(completeness_a))
    print("Correctness :", np.mean(correctness_a))
    print("Fidelity :", np.mean(fidelity_a))
    print("Robustness :", np.mean(robustness_a))


    with open(os.path.join("./json", name_json_rules), 'w') as f:   #enregistrement des regles dans un fichier json
        json.dump(listf, f, cls=NpEncoder)

if __name__ == "__main__":
    main()
