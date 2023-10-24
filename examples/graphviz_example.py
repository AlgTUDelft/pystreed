import graphviz
from sklearn.datasets import load_iris
from pystreed import STreeDClassifier
import pandas as pd

iris_data = load_iris(as_frame=True)
data = pd.DataFrame(iris_data["data"], columns=iris_data["feature_names"])
target = iris_data["target"]
class_names = iris_data["target_names"]

model = STreeDClassifier("accuracy", max_depth = 4, max_num_nodes=6, time_limit=100, verbose=True)
model.fit(data, target)

model.print_tree(label_names=class_names)

model.export_dot("tree.dot", label_names=class_names)


with open("tree.dot") as f:
    dot_graph = f.read()
g = graphviz.Source(dot_graph)
g.render('tree.dot', outfile="tree.pdf", view=True, cleanup=True)