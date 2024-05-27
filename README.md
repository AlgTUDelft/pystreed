[![CMake build](https://github.com/algtudelft/pystreed/actions/workflows/cmake.yml/badge.svg)](https://github.com/algtudelft/pystreed/actions/workflows/cmake.yml)
[![Pip install](https://github.com/algtudelft/pystreed/actions/workflows/pip.yml/badge.svg)](https://github.com/algtudelft/pystreed/actions/workflows/pip.yml)

# STreeD: Separable Trees with Dynamic programming
By: Jacobus G. M. van der Linden [(e-mail)](mailto:J.G.M.vanderLinden@tudelft.nl)

STreeD is a framework for optimal binary decision trees with _separable_ optimization tasks. A separable optimization task is a task that can be optimized separately for the left and right subtree. The current STreeD Framework implements a broad set of such optimization tasks, from group fairness constraints to survival analysis. For an explanation of each application, see below.
For details on what tasks are separable and how the algoritm works, see our paper.

If you use STreeD, please cite our paper:

* Van der Linden, Jacobus G. M., Mathijs M. de Weerdt, and Emir Demirović. "Necessary and Sufficient Conditions for Optimal Decision Trees using Dynamic Programming." _Advances in Neural Information Processing Systems_. 2023. [pdf](https://proceedings.neurips.cc/paper_files/paper/2023/file/1d5fce9627e15c84db572a66e029b1fc-Paper-Conference.pdf) 

## Python usage

### Install from PyPi
The `pystreed` python package can be installed from PyPi using `pip`:

```sh
pip install pystreed
```

### Install from source using pip
The `pystreed` python package can also be installed from source as follows:

```sh
git clone https://github.com/AlgTUDelft/pystreed.git
cd pystreed
pip install . 
```

### Example usage
`pystreed` can be used, for example, as follows:

```python
from pystreed import STreeDClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Read data
df = pd.read_csv("data/classification/anneal.csv", sep=" ", header=None)
X = df[df.columns[1:]]
y = df[0]

# Fit the model
model = STreeDClassifier(max_depth = 3, max_num_nodes=5)
model.fit(X, y)

model.print_tree()

yhat = model.predict(X)

accuracy = accuracy_score(y, yhat)
print(f"Train Accuracy Score: {accuracy * 100}%")
```


See the [examples](examples) folder for a number of example usages.

Note that some of the examples require the installation of extra python packages:

```sh
pip install matplotlib seaborn graphviz scikit-survival pydl8.5 pymurtree
```

Note that `pymurtree` is currently not available for pip install yet. It can be installed from [source](https://github.com/MurTree/pymurtree/) (install the `develop` branch)

Graphviz additionaly requires another instalation of a binary. See [their website](https://graphviz.org/download/).

## C++ usage

### Compiling
The code can be compiled on Windows or Linux by using cmake. For Windows users, cmake support can be installed as an extension of Visual Studio and then this repository can be imported as a CMake project.

For Linux users, they can use the following commands:

```sh
mkdir build
cd build
cmake ..
cmake --build .
```
The compiler must support the C++17 standard

### Running
After STreeD is built, the following command can be used (for example):
```sh
./STreeD -task accuracy -file ../data/cost-sensitive/car/car-train-1.csv -max-depth 3 -max-num-nodes 7
```

Run the program without any parameters to see a full list of the available parameters.

### Docker
Alternatively, docker can be used to build and run STreeD:
```
docker build -t streed .
docker container run -it streed /STreeD/build/STREED -task accuracy -file /STreeD/data/cost-sensitive/car/car-train-1.csv -max-depth 3 -max-num-nodes 7
```

## Applications
Currently, STreeD implements the following optimization tasks:

### Classification
`STreeDClassifier` implements the following optimization tasks:

* `accuracy`: Minimizes the misclassification score. 
* `cost-complex-accuracy`: Minimizes the misclassification score plus the cost for adding a branching node by the parameter `cost_complexity`.
* `f1-score`: Maximizes the F1-score.

See [examples/accuracy_example.py](examples/accuracy_example.py) for an example.

### Cost-Sensitive Classification
`STreeDCostSensitiveClassifier` implements a cost-sensitive classifier. Costs can both be attributed to features and misclassifications.

The costs can be specified with `CostSpecifier` object. This object is either initialized with a file name for the cost specification and the number of classes; or with the misclassification cost matrix, and the cost specifier per feature. When testing a feature in a branch node, the cost for that feature is paid for every instance that passes through it. When another feature from the same group was tested before, the discounted cost is paid. When another feature that is binarized from the same original feature is already tested, the cost is zero.

Note that currently `STreeDCostSensitiveClassifier` does not support automatic binarization.

See [examples/cost_sensitive_example.py](examples/cost_sensitive_example.py) for an example.

### Instance-Cost-Sensitive Classification
`STreeDInstanceCostSensitiveClassifier` implements an instance-cost-sensitive classifier. Each instance can  have a different misclassification cost per label.

The costs can be specified with a `CostVector` object. For each instance, initialize a `CostVector` object with a list of the costs for each possible label. 

See [examples/instance_cost_sensitive_example.py](examples/instance_cost_sensitive_example.py) for an example.

### Classification under a Group Fairness constraint
`STreeDGroupFairnessClassifier` implements a classifier that satisfies a group fairness constraint.
The maximum amount of discrimination on the training data can be specified by the `discrimination-limit` parameter, e.g., 0.01 for maximum of 1% discrimination.

Currently two fairness constraint optmization tasks are implemented:
* `group-fairness`: This satisfies a _demographic parity_ constraint. Demographic parity requires that the positive rates for both groups is equal.
* `equality-of-opportunity`: This satisfies a _equality of opportunity_ constraint. Equality of opportunity requires that the _true_ positive rates for both groups is equal.

Note:
1. `STreeDGroupFairnessClassifier` assumes binary classification (only two labels: positive = 1, negative = 0).
2. `STreeDGroupFairnessClassifier` assumes that the first binary feature column is the discrimination-sensitive feature. Otherwise the sensitive feature can be specified with the `sensitive_feature` parameter.

See [examples/group_fair_example.py](examples/group_fair_example.py) for an example.

### Prescriptive policy generation
`STreeDPrescriptivePolicyGenerator` implements a policy generation solver. Counterfactual scores need to be provided. The current implementation allows for three different teacher methods, as specified by the `teacher_method` parameter:
* `DM`: the _direct method_ or _Regress & Compare_ method. This teacher specifies for every treatment (label) what the expected outcome is.
* `IPW`: the _inverse propensity weighting_ method. This teacher provides the propensity scores mu(x, k): the probability of treatment k happening for feature vector x.
* `DR`: the _doubly robust_ method: a combination of the direct method and the inverse propensity weighting method.

The teacher data needs to be passed to the solver by initializing a `PPGData` object for every instance. The PPGData initializer expects the following parameters:
* `historic_treatment : int` : the historic treatment label
* `historic_outcome : float` : the historic outcome
* `propensity_score : float` : the propensity score for the historic treatment
* `predicted_outcome : List[float]` : the _regress & compare_ prediction for each possible treatment
Optional (for testing)
* `optimal_treatment : int` : the optimal treatment
* `counterfactual_outcome : List[float]` : the counterfactual outcome 

Only the data which will be used by the teacher method needs to be specified, the rest can be initialized with zero's.

See [examples/prescriptive_policy_example.py](examples/prescriptive_policy_example.py) for an example.

### Regression

`STreeDRegressor` implements two variants of regression, as specified by the optimization task parameter
* `regression`: Miminimizes the _sum of squared errors_.
* `cost-complex-regression`: Minimizes the _sum of squared errors_ plus the cost for adding a branching node by the parameter `cost_complexity`. For runtime improvement, custom lower bounds can be specified if `use_task_lower_bound=True`. The custom lower bound `regression_bound` can be set to either `"equivalent"` to use the equivalent-points bound or `"kmeans"` to use a k-means lower bound.

See [examples/regression_example.py](examples/regression_example.py) for an example.

If you use STreeD for _regression_, please cite our paper:

* Van den Bos, M., Jacobus G. M. van der Linden, and Emir Demirović. "Piecewise Constant and Linear Regression Trees: An Optimal Dynamic Programming Approach." In _Proceedings of ICML-24_, 2024.

### Piecewise Linear Regression
`STreeDPiecewiseLinearRegressor` implements a solver for optimizing piecewise linear regression trees, with a linear elastic net regression predictor in every leaf node. The lasso and ridge penalization can be set with the `lasso_penalty` and `ridge_penalty` and parameters. The addition of a new branching node is penalized by the `cost_complexity` parameter. Alternatively, `STreeDPiecewiseLinearRegressor` can learn a simple linear regression model in every leaf by setting `simple = True`. The simple linear regression model is penalized only with the ridge penalization.

`STreeDPiecewiseLinearRegressor` only uses the continuous features for fitting the linear lasso regression model in every leaf node. These continuous features can be automatically inferred from the data or explicitly specified using the `continuous_columns` parameter of the `fit` method.

To prevent fitting linear models on too little data, `STreeDPiecewiseLinearRegressor` by default sets the `min_leaf_node_size` parameter to at least 5 times the number of continuous features or to at least 5 when fitting a simple linear regression model.

See [examples/piecewise_linear_regression_example.py](examples/piecewise_linear_regression_example.py) for an example.

If you use STreeD for _piecewise linear regression_, please cite our paper:

* Van den Bos, M., Jacobus G. M. van der Linden, and Emir Demirović. "Piecewise Constant and Linear Regression Trees: An Optimal Dynamic Programming Approach." In _Proceedings of ICML-24_, 2024.

### Survival analysis
`STreeDSurvivalAnalysis` implements an optimal survival tree method, by optimizing the proportional hazard function of LeBlanc and Crowly, "Relative Risk for Censored Survival Data," _Biometrics_ 48.2 (1992): 411-425. Each leaf node predicts a risk factor $\theta$ which is used to shift the base hazard model $\hat{\Lambda}(t)$.  The Nelson-Aalen estimator is used as a stepwise survival function $\hat{S}(t) = e^{-\theta \hat{\Lambda}(t)}$.

Instead of a label, the input data expects a two-dimensional array with for each instance 1) a binary censoring indicator and 2) a time-of-event (death or censoring).

See [examples/survival_analysis_example.py](examples/survival_analysis_example.py) for an example.

If you use STreeD for _survival analysis_, please cite our paper:

* Huisman, T., Jacobus G. M. van der Linden, and Emir Demirović. "Optimal Survival Trees: A Dynamic Programming Approach." _Proceedings of AAAI-24_. 2024. [pdf](https://arxiv.org/pdf/2401.04489.pdf)

## Parameters
STreeD can be configured by the following parameters:
* `max_depth` : The maximum depth of the tree. Note that a tree of depth zero has a single leaf node. A tree of depth one has one branching node and two leaf nodes.
* `max_num_nodes` : The maximum number of _branching_ nodes in the tree.
* `min_leaf_node_size` : The minimum number of samples required in each leaf node.
* `time_limit` : The run time limit in seconds. If the time limit is exceeded a possibly non-optimal tree is returned.
* `feature_ordering` : The order in which the features are considered for branching. Default is `"gini"` which sorts the features by gini-impurity decrease. The alternative (and default for survival analysis) is `"in-order"` which considers the feature in order of appearance.
* `hyper_tune` : Use STreeD's special hyper-tune method.
* `use_branch_caching` : Enables or disables the use of branch caching.
* `use_dataset_caching` : Enables or disables the use of dataset caching.
* `use_terminal_solver` : Enables or disables the use of the special solver for trees of depth two.
* `use_similarity_lower_bound` : Enables or disables the use of the similarity lower bound.
* `use_upper_bound` : Enables or disables the use of upper bounds.
* `use_lower_bound` : Enables or disables the use of lower bounds.
* `verbose` : Enable or disable verbose output.
* `random_seed` : The random seed.

## Binarization
STreeD provides optimal decision trees for a given binarization. To help with the binarization, the `pystreed` package provides automatic binarization of categorical and continuous features.

Categorical features can be specified in the `fit` method by using the `categorical_columns` parameter. These features are binarized using one-hot encoding. The maximum number of categories per categorical feature is specified with the `n_categories` parameter. If the categorical feature exceeds this number, the `n_categories - 1` most common categories are encoded with one binary feature each, and all other categories are encoded with an 'other' category.

Continuous features are automatically recognized by STreeD. Each continuous feature is binarized into a number of binary features as specified by the `n_thresholds` parameter. Each binary feature has the form `x <= t`, with `x` the continuous feature and `t` the threshold.
 Currently STreeD provides three ways of automatically binarizing continuous features, as specified by the `continuous_binarize_strategy` parameter:
* `uniform` : Select thresholds uniformly from the interval `[min(x), max(x)]`
* `quantile` : Select thresholds uniformly from the quantile distribution
* `tree` : Train a CART tree using only one continuous feature with up to `n_thresholds` branching nodes, and select the thresholds from the branching nodes.

Note that STreeD provides an optimal decision tree for the given binarization. The binarization should therefore be chosen with care.

See [examples/binarize_example.py](examples/binarize_example.py) for an example.

## Overfitting and tuning

To prevent overfitting the size of the tree can be tuned. This can be done in the standard way using `scikit-learn` methods, see [examples/gridsearch_example.py](examples/gridsearch_example.py). 

For improving runtime performance, STreeD also provides a custom tuning method that in some cases can reuse the cache from previous runs.
To use this, initialize the STreeD model with `hyper_tune=True`.

STreeD's default method of hypertuning directly tunes the depth and the number of branching nodes using five-fold cross validation. Some optimization tasks specify their custom tuning method, such as `cost-complex-accuracy` which tunes the `cost_complexity` parameter.

## Miscellaneous 
* STreeD assumes classification labels are in the range `0 ... n_labels - 1`. Not meeting this assumption may influence the algorithm's performance. Use sklearn's `LabelEncoder` to prevent this.

## References
This work is a continuation of the following previous papers (with corresponding repositories)

1. Demirović, Emir, et al. "Murtree: Optimal decision trees via dynamic programming and search." _Journal of Machine Learning Research_ 23.26 (2022): 1-47. [pdf](https://www.jmlr.org/papers/volume23/20-520/20-520.pdf) / [source](https://bitbucket.org/EmirD/murtree/src/master/)
2. Demirović, Emir, and Peter J. Stuckey. "Optimal decision trees for nonlinear metrics." _Proceedings of the AAAI conference on artificial intelligence._ Vol. 35. No. 5. 2021. [pdf](https://ojs.aaai.org/index.php/AAAI/article/download/16490/16297) / [source](https://bitbucket.org/EmirD/murtree-bi-objective/src/master/)
3. Van der Linden, Jacobus G. M., Mathijs M. de Weerdt, and Emir Demirović. "Fair and Optimal Decision Trees: A Dynamic Programming Approach." _Advances in Neural Information Processing Systems_. 2022. [pdf](https://openreview.net/pdf?id=LCIZmSw1DuE) / [source](https://gitlab.tudelft.nl/jgmvanderlinde/dpf)
