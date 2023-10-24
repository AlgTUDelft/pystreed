from pystreed import STreeDPrescriptivePolicyGenerator
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Read data
df_train = pd.read_csv("data/prescriptive-policy/warfarin/train_rf_dt_0.06_0_0.csv", sep=" ", header=None)
X = df_train[df_train.columns[11:]].values
y = df_train[df_train.columns[1:11]].values

df_test = pd.read_csv("data/prescriptive-policy/warfarin/test_rf_dt_0.06_0_0.csv", sep=" ", header=None)
X_test = df_test[df_test.columns[11:]].values
y_test = df_test[df_test.columns[1:11]].values
"""
Column description in this file:
Training data
Column  0           the label [historic treatment]
Column  1           historic treatment
Column  2           historic outcome
Column  3           propensity score
Column  4, 5, 6     regress & compare yhat prediction
Test data (optional)
Column  7           the optimal treatment
Column  8, 9, 10    the (counterfactual) outcome y
"""

# Fit the model
model = STreeDPrescriptivePolicyGenerator(max_depth = 4, teacher_method="IPW", time_limit=100, verbose=True)
model.fit(X,y)

model.print_tree()

yopt = y_test[:, 6]
yhat = model.predict(X_test)

accuracy = accuracy_score(yopt, yhat)
print(f"Test Accuracy Score: {accuracy * 100}%")
print(f"Test Average Outcome: {model.score(X_test, y_test)}")