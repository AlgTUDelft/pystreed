from pystreed import STreeDSurvivalAnalysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored

# Read the data
df = pd.read_csv("data/survival-analysis/acath_binary.txt", sep=" ", header=None)
X = df[df.columns[2:]].values
y = df[[1,0]].to_records(index=False, column_dtypes={0: np.double, 1: bool})
events = y["1"]
times = y["0"]

# Show the Kaplan-Meier estimator of the original data
time, survival_prob, conf_int = kaplan_meier_estimator( events, times, conf_type="log-log")
plt.step(time, survival_prob, where="post")
plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
plt.ylim(0, 1)
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.show()

# Train an optimal survival tree model
model = STreeDSurvivalAnalysis(max_depth = 3, max_num_nodes=4)
model.fit(X, y)

# plot the survival function for the first five instances
pred_surv = model.predict_survival_function(X[:5])
time_points = np.arange(1, 400)
for i, surv_func in enumerate(pred_surv):
    plt.step(time_points, surv_func(time_points), where="post", label=f"Sample {i + 1}")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")
plt.show()

# Measure the performance of the model
prediction = model.predict(X)
result = concordance_index_censored(events, times, prediction)
print("Harrell's concordance index: ", result[0])
print("Objective score: ", model.score(X, y))

# Export the tree as pdf
model.export_dot("tree.dot")

import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
g = graphviz.Source(dot_graph)
g.render('tree.dot', outfile="tree.pdf", view=True, cleanup=True)
