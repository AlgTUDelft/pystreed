from pystreed import STreeDClassifier
import pandas as pd
from warnings import simplefilter
simplefilter(action='ignore', category=UserWarning)

# Read data

df = pd.read_csv("data/classification/segment.csv", sep=" ", header=None)
X = df[df.columns[1:]].values
y = df[0].values

# Fit the model
for max_depth in range(0, 5):
    model = STreeDClassifier(max_depth = max_depth, upper_bound=0, time_limit=100, verbose=False)
    model.fit(X,y)
    if model.is_fitted(): break
    print(f"No perfect tree for d = {max_depth}")


for max_num_nodes in range(0, 2**max_depth):
    model = STreeDClassifier(max_depth = max_depth, max_num_nodes=max_num_nodes, upper_bound=0, time_limit=100, verbose=False)
    model.fit(X,y)
    if model.is_fitted(): break
    print(f"No perfect tree for d = {max_depth}, n = {max_num_nodes}")


print(f"\nSmallest perfect tree with d = {max_depth}, n = {max_num_nodes}")

print("")

model.print_tree()
