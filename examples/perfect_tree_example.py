from pystreed import STreeDClassifier
import pandas as pd
import time
from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)

# Read data

df = pd.read_csv("data/classification/segment.csv", sep=" ", header=None)
X = df[df.columns[1:]].values
y = df[0].values

# Fit the model
max_depth = 20
model = STreeDClassifier("cost-complex-accuracy", max_depth = max_depth, cost_complexity = 1.0 / (max_depth*len(df)), time_limit=100, verbose=False)
start = time.perf_counter()
model.fit(X,y)
duration = time.perf_counter() - start

# repor the score
accuracy = model.score(X,y)
depth = model.get_depth()
num_nodes = model.get_n_leaves() - 1

if accuracy == 1.0:
    print(f"\nSmallest perfect tree with d = {depth}, n = {num_nodes} in {duration:.2f} seconds")
else:
    print(f"\nFound tree with accuracy = {accuracy*100}% misclassifications with d = {depth}, n = {num_nodes} in {duration:.2f} seconds")

# print the tree
print("")
model.print_tree()

