from pystreed import STreeDCostSensitiveClassifier
from pystreed.data import CostSpecifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Specify the costs
cost_specifier = CostSpecifier("data/cost-sensitive/car/car-middle-costs.csv", 4) # Number of labels = 4

accuracy_list = []
costs_list = []

for i in range(1, 101):
    # Read data
    df_train = pd.read_csv(f"data/cost-sensitive/car/car-train-{i}.csv", sep=" ", header=None)
    X_train = df_train[df_train.columns[1:]].values
    y_train = df_train[0].values

    df_test = pd.read_csv(f"data/cost-sensitive/car/car-test-{i}.csv", sep=" ", header=None)
    X_test = df_test[df_test.columns[1:]].values
    y_test = df_test[0].values

    # Fit the model
    model = STreeDCostSensitiveClassifier(max_depth = 3)
    model.fit(X_train, y_train, cost_specifier)

    yhat = model.predict(X_test)
    accuracy = accuracy_score(y_test, yhat)
    print(f"{i:3d} | Test Accuracy Score: {accuracy * 100:.1f}%")
    costs = model.score(X_test, y_test)
    print(f"{i:3d} | Total test costs:    {costs * 100:.2f}")

    accuracy_list.append(accuracy)
    costs_list.append(costs)

print(f"Mean accuracy: {sum(accuracy_list) / len(accuracy_list) * 100:.2f}%, Mean costs: {sum(costs_list) / len(costs_list) * 100:.1f}")