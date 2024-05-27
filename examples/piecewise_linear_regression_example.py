from pystreed import STreeDPiecewiseLinearRegressor, STreeDRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

raw_data = load_diabetes(as_frame=True)
X = raw_data["data"]
y = raw_data["target"]

# Piecewise linear Regression Tree
pwl_model = STreeDPiecewiseLinearRegressor(max_depth = 3, lasso_penalty=0.05, verbose=True)
pwl_model.fit(X, y)
yhat = pwl_model.predict(X)

pwl_r2 = r2_score(y, yhat)
pwl_mse = mean_squared_error(y, yhat)

# Piecewise constant Regression Tree
pwc_model = STreeDRegressor(max_depth = 3)
pwc_model.fit(X, y)
yhat = pwc_model.predict(X)

pwc_r2 = r2_score(y, yhat)
pwc_mse = mean_squared_error(y, yhat)

# Linear regression model
lr_model = LinearRegression()
lr_model.fit(X, y)
yhat = lr_model.predict(X)

lr_r2 = r2_score(y, yhat)
lr_mse = mean_squared_error(y, yhat)

print(f"PWL regressor (d=3) Train R2 Score: {pwl_r2}")
print(f"PWL regressor (d=3) Train MSE Score: {pwl_mse}")

print(f"PWC regressor (d=3) Train R2 Score: {pwc_r2}")
print(f"PWC regressor (d=3) Train MSE Score: {pwc_mse}")

print(f"LR regressor Train R2 Score: {lr_r2}")
print(f"LR regressor Train MSE Score: {lr_mse}")