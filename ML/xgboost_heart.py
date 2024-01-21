import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score

"""
PRECISION XGBOOST SUR HEART 80.33%
{'colsample_bytree': 0.3, 'gamma': 0.1, 'learning_rate': 0.3, 'max_depth': 10, 'min_child_weight': 1}
"""


df_x = pd.read_csv("Data/heart/heart.csv", delimiter=' ')
df_y = pd.read_csv("Data/heart/labels_heart.csv")
df = pd.concat([df_x, df_y], axis=1)

Y = df["output"]
X = df.drop(["output"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]   
}

"""XGB_model = xgb.XGBClassifier()
xgb_cv_model  = GridSearchCV(XGB_model,params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train, y_train)
print(xgb_cv_model.best_params_)
model = xgb.XGBClassifier(**xgb_cv_model.best_params_).fit(X_train, y_train)"""
model = xgb.XGBClassifier(colsample_bytree=0.3, gamma=0.1, learning_rate=0.3, max_depth=10, min_child_weight=1)
score = cross_val_score(model, X, Y, cv=10)
score.mean()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))