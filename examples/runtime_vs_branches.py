from pystreed import STreeDClassifier
import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys
try:
    # Obtain branches from https://github.com/Chaoukia/branches
    # Chaouki, A., Read, J., & Bifet, A. (2024). Branches: A Fast Dynamic Programming and Branch 
    # & Bound Algorithm for Optimal Decision Trees. arXiv preprint arXiv:2406.02175.
    branches_path = "<fill_in_path>"
    sys.path.append(f"{branches_path}\\src")
    from branches import Branches
except:
    print("Error: provide the import directory for Branches")
    sys.exit(1)

datasets = ["vote",
            "segment"]
scores = []

for dataset in datasets:

    # Read data
    df = pd.read_csv(f"data/classification/{dataset}.csv", sep=" ", header=None)
    X = df[df.columns[1:]].values
    y = df[0].values

    max_depth = 8
    lamda = 0.01
    for i in range(5):
        # Fit the STreeD model
        streed_model = STreeDClassifier("cost-complex-accuracy", max_depth = max_depth, cost_complexity=lamda)
        start = time.perf_counter()
        streed_model.fit(X, y)
        streed_duration = time.perf_counter() - start
        streed_score = round((1 - streed_model.fit_result.score()) * len(X))
        print(f"{dataset.capitalize():10s} Run {i+1:2d}  |  STreeD    |  run time: {streed_duration:.3f}  | score: {streed_score}")
        scores.append({"Method": "STreeD", "Dataset": dataset, "Runtime": streed_duration})

        # Fit the Branches model
        br_data = np.c_[X,y]
        br_alg = Branches( br_data , encoding="binary")
        br_alg.reinitialise()
        start = time.perf_counter()
        br_alg.solve(lamda, n=1e10, time_limit=600, print_iter=1e9)
        branches_duration = time.perf_counter() - start
        branches_score = ((br_alg.predict(br_data[:, :-1]) != br_data[:, -1]).sum())#/br_alg.n_total
        print(f"{dataset.capitalize():10s} Run {i+1:2d}  |  Branches  |  run time: {branches_duration:.3f} | score: {branches_score}")
        scores.append({"Method": "Branches", "Dataset": dataset, "Runtime": branches_duration})

results = pd.DataFrame(scores)

sns.barplot(results, x="Dataset", y="Runtime", hue="Method")
plt.ylabel("Runtime (s)")
plt.xlabel("Data set")

plt.show()