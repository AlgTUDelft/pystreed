from pystreed import STreeDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time


df = pd.read_csv("data/classification/anneal.csv", sep=" ", header=None)
X = df[df.columns[1:]].values
y = df[0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##################################################################
##### 1. Tune using GridSearchCV from sklearn ####################
##################################################################

model = STreeDClassifier(max_depth = 4)

params = [{'optimization_task': ["cost-complex-accuracy"], 'cost_complexity': [0.0001, 0.0005, 0.001, 0.005, 0.01]},
         {'optimization_task': ["accuracy"], 'max_num_nodes': list(range(1,16))}]
params = [{'optimization_task': ["accuracy"], 'max_num_nodes': list(range(1,16))}]

gs_knn = GridSearchCV(model,
                      param_grid=params,
                      scoring='accuracy',
                      cv=5,
                      n_jobs=1,
                      verbose=3)
start = time.perf_counter()
gs_knn.fit(X_train, y_train)
gs_duration = time.perf_counter() - start

print(f"\nSklearn gridsearch finished in {gs_duration} seconds")
print("Best params from grid search: ", gs_knn.best_params_)

yhat = gs_knn.predict(X_test)

accuracy = accuracy_score(y_test, yhat)
print(f"Test Accuracy Score: {accuracy * 100}%")

##################################################################
##### 2. Tune using the hyper_tune parameter of STreeD ###########
##################################################################

model = STreeDClassifier(max_depth = 4, hyper_tune=True, verbose=True)

start = time.perf_counter()
model.fit(X_train, y_train)
ht_duration = time.perf_counter() - start

print(f"\nSTreeD hyper-tune finished in {ht_duration} seconds")
params = model.get_solver_params()
print(f"Best parameters: max-depth = {params.max_depth}, max-num-nodes = {params.max_num_nodes}")

yhat = model.predict(X_test)

accuracy = accuracy_score(y_test, yhat)
print(f"Test Accuracy Score: {accuracy * 100}%")

##################################################################
##### Conclusion #################################################
##################################################################
#
# For some optimization problems, STreeD can reuse its cache to 
# improve hyper-tuning runtime performance
#
##################################################################