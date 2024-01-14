import pandas as pd

dfx = pd.read_csv('Data/Covid-19/Covid-19.csv', sep=",")
dfy = pd.read_csv('Data/Covid-19/labels_Covid-19.csv', sep=",")


df = pd.concat([dfx, dfy], axis=1).sample(frac=1)


x = df[["Fever", "Tiredness", "Dry-Cough","Difficulty-in-Breathing","Sore-Throat","Pains","Nasal-Congestion","Runny-Nose","Diarrhea","None_Experiencing","Age_0-9","Age_10-19","Age_20-24","Age_25-59","Age_60+","Gender_Female","Gender_Male","Contact_Dont-Know","Contact_No","Contact_Yes","Symptoms_Score"]].copy()
y = df[["Condition"]]

x.to_csv("Data/Covid-19/Covid-19.csv", index=False, sep=" ")
y.to_csv("Data/Covid-19/labels_Covid-19.csv", index=False, sep=" ")