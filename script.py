import pandas as pd  # melanger les donnees

dfx = pd.read_csv('Data/heart/heart.csv', sep=" ")
dfy = pd.read_csv('Data/heart/labels_heart.csv', sep=" ")


df = pd.concat([dfx, dfy], axis=1).sample(frac=1)

x = df[["age", "sex", "cp", "trtbps", "chol", "fbs", "restecg", "thalachh", "exng", "oldpeak", "slp", "caa", "thall"]].copy()
y = df[["output"]]

x.to_csv("Data/heart/heart.csv", index=False, sep=" ")
y.to_csv("Data/heart/labels_heart.csv", index=False, sep=" ")