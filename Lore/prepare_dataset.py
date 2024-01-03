from util import *

def prepare_diabete_dataset(filename, label_name, path_data):
    # Read Dataset
    valeur = pd.read_csv(path_data + filename, delimiter=' ', skipinitialspace=True)
    label = pd.read_csv(path_data + label_name)
    df = pd.concat([valeur, label], axis=1)
    #columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    columns = df.columns
    df = df[columns]
    columns = df.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    df = df[columns]
    class_name = 'Outcome'
    possible_outcomes = list(df[class_name].unique())

    type_features, features_type = recognize_features_type(df, class_name)

    discrete = []
    discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete=discrete,
                                                   continuous=None)

    columns_tmp = list(columns)
    columns_tmp.remove(class_name)
    idx_features = {i: col for i, col in enumerate(columns_tmp)}

    # Dataset Preparation for Scikit Alorithms
    df_le, label_encoder = label_encode(df, discrete)
    X = df_le.loc[:, df_le.columns != class_name].values
    y = df_le[class_name].values

    dataset = {
        'name': filename.replace('.csv', ''),
        'df': df,
        'columns': list(columns),
        'class_name': class_name,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete,
        'continuous': continuous,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'X': X,
        'y': y,
    }

    return dataset