import graphviz
from sklearn.datasets import load_diabetes
from pystreed import STreeDPiecewiseLinearRegressor
import pandas as pd

data = load_diabetes(as_frame=True)
X = pd.DataFrame(data["data"], columns=data["feature_names"])
y = data["target"]

model = STreeDPiecewiseLinearRegressor(simple=True, max_depth = 2)
model.fit(X, y)

model.print_tree()

model.export_dot("tree.dot")


with open("tree.dot") as f:
    dot_graph = f.read()
g = graphviz.Source(dot_graph)
g.render('tree.dot', outfile="tree.pdf", view=True, cleanup=True)