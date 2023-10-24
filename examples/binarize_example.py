import graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from pystreed import STreeDClassifier
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

iris_data = load_iris(as_frame=True)
data = iris_data["data"]
target = iris_data["target"]
class_names = iris_data["target_names"]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

model = STreeDClassifier("accuracy", max_depth = 3, time_limit=100)

params = [{'continuous_binarize_strategy': ["tree", "quantile", "uniform"],
           "n_thresholds": list(range(1, 10)),
           "max_num_nodes": list(range(8))}]

gs_knn = GridSearchCV(model,
                      param_grid=params,
                      scoring='accuracy',
                      cv=5,
                      n_jobs=1,
                      verbose=3)

gs_knn.fit(X_train, y_train)

print("Best params from grid search: ", gs_knn.best_params_)


yhat = gs_knn.predict(X_test)

accuracy = accuracy_score(y_test, yhat)
print(f"Test Accuracy Score: {accuracy * 100}%")

gs_knn.best_estimator_.print_tree(label_names=class_names)

gs_knn.best_estimator_.export_dot("tree.dot", label_names=class_names)

with open("tree.dot") as f:
    dot_graph = f.read()
g = graphviz.Source(dot_graph)
g.render('tree.dot', outfile="tree.pdf", view=True, cleanup=True)