from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pystreed import STreeDRegressor
from warnings import simplefilter
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
simplefilter(action='ignore', category=FutureWarning)

def fetch_servo():
    url = "https://archive.ics.uci.edu/static/public/87/servo.zip"
    r = urlopen(url).read()
    file = ZipFile(BytesIO(r))
    csv = file.open("servo.data")
    names = ["motor", "screw", "pgain", "vgain", "class"]
    df = pd.read_csv(csv, sep=",", header=None, names=names)
    sorted_names = names[-1:] + names[:-1]  # Bring label to front
    df = df.reindex(columns=sorted_names)
    return df

df = fetch_servo()
target = df["class"]
data = df[df.columns[1:]]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

model = STreeDRegressor(max_depth = 3, n_categories=3)
model.fit(X_train, y_train)


yhat = model.predict(X_test)

mse = mean_squared_error(y_test, yhat)
print(f"Test MSE Score: {mse}\n")

model.print_tree()

