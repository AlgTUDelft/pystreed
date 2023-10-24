from pystreed import STreeDClassifier
from pydl85 import DL85Classifier
import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pymurtree import OptimalDecisionTreeClassifier as MurTreeClassifier

datasets = ["anneal", "diabetes", "hepatitis"]
scores = []

for dataset in datasets:

    # Read data
    df = pd.read_csv(f"data/classification/{dataset}.csv", sep=" ", header=None)
    X = df[df.columns[1:]].values
    y = df[0].values

    for d in range(1, 5):

        for i in range(5):
            # Fit the STreeD model
            streed_model = STreeDClassifier(max_depth = d)
            start = time.perf_counter()
            streed_model.fit(X, y)
            streed_duration = time.perf_counter() - start
            print(f"{dataset.capitalize():10s} Run {i+1:2d}  |  STreeD  d={d}  |  run time: {streed_duration:.3f}")

            # Fit the dl8.5 model
            dl85_model = DL85Classifier(max_depth=d)
            start = time.perf_counter()
            dl85_model.fit(X,y)
            dl85_duration = time.perf_counter() - start
            print(f"{dataset.capitalize():10s} Run {i+1:2d}  |  DL8.5   d={d}  |  run time: {dl85_duration:.3f}")

            # Fit the MurTree model
            murtree_model = MurTreeClassifier(max_depth=d, max_num_nodes=2**d-1, verbose=False)
            start = time.perf_counter()
            murtree_model.fit(X,y)
            murtree_duration = time.perf_counter() - start
            print(f"{dataset.capitalize():10s} Run {i+1:2d}  |  MurTree d={d}  |  run time: {murtree_duration:.3f}")

            # All optimal models, so should have precisely the same training score
            assert(murtree_model.score() == round((1 - streed_model.fit_result.score()) * len(X)))
            assert(murtree_model.score() == round((1-dl85_model.accuracy_) * len(X)))

            scores.append({"Method": "STreeD", "Dataset": dataset, "Depth": d, "Runtime": streed_duration})
            scores.append({"Method": "DL8.5",  "Dataset": dataset, "Depth": d, "Runtime": dl85_duration})
            scores.append({"Method": "MurTree",  "Dataset": dataset, "Depth": d, "Runtime": murtree_duration})

results = pd.DataFrame(scores)

means = results.groupby(["Method", "Depth", "Dataset"]).mean()["Runtime"].unstack("Method")
print(means)

g = sns.relplot(
    data=results,
    x="Depth", y="Runtime",
    hue="Method", style="Method",
    col="Dataset",
    kind="line",
    facet_kws=dict(sharey=False),
)

g.axes.flat[0].xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()