import pandas as pd
from pystreed import STreeDGroupFairnessClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/fairness/student-mat-binarized.csv", header=None, sep=" ")
y = df.pop(0)
X = df
X.columns = range(X.columns.size)

print(X.columns)

cls = STreeDGroupFairnessClassifier(max_depth=2, discrimination_limit=0.05)
cls.fit(X, y)

yhat = cls.predict(X)

print(f"Train Accuracy Score: {accuracy_score(yhat, y) * 100}%")

group0 = X[0] == 0
group1 = X[0] == 1
group0_size = sum(group0)
group1_size = sum(group1)

group0_score = sum(yhat[group0]) / group0_size
group1_score = sum(yhat[group1]) / group1_size

print(f"Score group 0: {group0_score * 100:.2f}")
print(f"Score group 1: {group1_score * 100:.2f}")
print(f"Discrimination: {(group1_score - group0_score) * 100:.2f}%")

