import lore
import torch
from Anchor.model import MLP
from Anchor.prepare_dataset import *
from Anchor.neighbor_generator import *

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def main():
    path_data = 'Data/diabetes/'
    dataset_name = 'diabetes.csv'
    class_name = 'labels_diabetes.csv'
    dataset = prepare_diabete_dataset(dataset_name, class_name, path_data)

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    blackbox = MLP(0).to(device)
    blackbox.load_state_dict(torch.load("./state_dict_model.pt", map_location=torch.device('cpu')))
    blackbox.eval()

    X2E = X_test
    y2E = blackbox.predict(X2E, device)
    y2E = np.asarray([dataset['possible_outcomes'][i] for i in y2E])

    idx_record2explain = 0

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
    for delta in explanation[1]:
        print('delta', delta)

    covered = lore.get_covered(explanation[0][1], dfX2E, dataset)
    print(len(covered))
    print(covered)

    print(explanation[0][0][dataset['class_name']], '<<<<')

    def eval(x, y):
        return 1 if x == y else 0

    precision = [1-eval(v, explanation[0][0][dataset['class_name']]) for v in y2E[covered]]
    print(precision)
    print(np.mean(precision), np.std(precision))


if __name__ == "__main__":
    main()
