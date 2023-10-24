from pystreed import STreeDClassifier
import pandas as pd

# Read data

df = pd.read_csv("data/classification/anneal.csv", sep=" ", header=None)
X = df[df.columns[1:]].values
y = df[0].values

# Fit the model
model = STreeDClassifier(max_depth = 3, max_num_nodes=5, time_limit=100, verbose=True)
model.fit(X,y)

# Evaluate the model
print(f"Train Accuracy Score: {model.score(X, y) * 100}%")