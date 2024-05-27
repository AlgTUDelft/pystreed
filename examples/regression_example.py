from pystreed import STreeDRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

df = pd.read_csv("data/regression/airfoil.csv", sep=" ", header=None)
X = df[df.columns[1:]]
y = df[0]

model = STreeDRegressor(max_depth = 3, verbose=True)
model.fit(X, y)

yhat = model.predict(X)

r2 = r2_score(y, yhat)
mse = mean_squared_error(y, yhat)
print(f"Train R2 Score: {r2}")
print(f"Train MSE Score: {mse}")