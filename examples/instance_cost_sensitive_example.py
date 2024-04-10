from pystreed import STreeDInstanceCostSensitiveClassifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Read the data
df = pd.read_csv("data/instance-cost-sensitive/maxsat21-32f.txt", sep=" ", header=None)
# In this file, 
# Column 0 is the optimal label (with lowest cost)
# Columns 1..8 are the costs of assigning the labels 0...7
# Columns 9... are the binary features
X = df[df.columns[9:]].values
y = df[0].values
costs = df[df.columns[1:9]].values

X_train, X_test, y_train, y_test, costs_train, costs_test = train_test_split(X, y, costs, test_size=0.20, random_state=42)

# Fit the model by passing the cost vector
model = STreeDInstanceCostSensitiveClassifier(max_depth = 5, time_limit=600, verbose=True)
model.fit(X_train, costs_train)

model.print_tree()

# Obtain the test predictions
yhat = model.predict(X_test)

# Obtain the test accuracy
print(f"Test Accuracy Score: {accuracy_score(y_test, yhat) * 100}%")

# Obtain the classification costs on the test set through model.score
print(f"Test Average Outcome: {model.score(X_test, costs_test)}")