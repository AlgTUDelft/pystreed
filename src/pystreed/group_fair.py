from pystreed.base import BaseSTreeDSolver
from typing import Optional, Union
from sklearn.utils._param_validation import Interval
from pystreed.utils import _color_brew
import numpy as np
import pandas as pd
import numbers

class STreeDGroupFairnessClassifier(BaseSTreeDSolver):

    _parameter_constraints: dict = {**BaseSTreeDSolver._parameter_constraints, 
        "discrimination_limit": [Interval(numbers.Real, 0, 1, closed="both")]
    }

    def __init__(self, 
                 optimization_task : str = 'group-fairness',
                 sensitive_feature : Union[int, str] = 0,
                 discrimination_limit : float = 0.05,
                 max_depth : int = 3,
                 max_num_nodes : Optional[int] = None,
                 min_leaf_node_size: int = 1,
                 time_limit : float = 600,
                 feature_ordering : str = "gini", 
                 hyper_tune: bool = False,
                 use_branch_caching: bool = True,
                 use_terminal_solver: bool = True,
                 use_similarity_lower_bound: bool = True,
                 use_upper_bound: bool = True,
                 use_lower_bound: bool = True,
                 verbose : bool = False,
                 random_seed: int = 27, 
                 continuous_binarize_strategy: str = 'quantile',
                 n_thresholds: int = 5,
                 n_categories: int = 5):
        """
        Construct a STreeDGroupFairnessClassifier

        Parameters:
            optimization_task: the objective used for optimization.
            sensitive_feature: str or int, the name or index of the sensitive feature (default is index 0)
            max_depth: the maximum depth of the tree
            max_num_nodes: the maximum number of branching nodes of the tree
            min_leaf_node_size: the minimum number of training instance that should end up in every leaf node
            time_limit: the time limit in seconds for fitting the tree
            feature_ordering: heuristic for the order that features are checked. Default: "gini", alternative: "in-order": the order in the given data
            hyper_tune: Use five-fold validation to tune the size of the tree to prevent overfitting
            use_branch_caching: Enable/Disable branch caching.
            use_terminal_solver: Enable/Disable the depth-two solver (Enabled typically results in a large runtime advantage)
            use_similarity_lower_bound: Enable/Disable the similarity lower bound (Enabled typically results in a large runtime advantage)
            use_upper_bound: Enable/Disable the use of upper bounds (Enabled is typically faster)
            use_lower_bound: Enable/Disable the use of lower bounds (Enabled is typically faster)
            verbose: Enable/Disable verbose output
            random_seed: the random seed used by the solver (for example when creating folds)
            continuous_binarization_strategy: the strategy used for binarizing continuous features
            n_thresholds: the number of thresholds to use per continuous feature
            n_categories: the number of categories to use per categorical feature
        """

        if not optimization_task in ["group-fairness", "equality-of-opportunity"]:
            raise ValueError(f"Invalid value for optimization_task: {optimization_task}")
        BaseSTreeDSolver.__init__(self, optimization_task, 
            max_depth=max_depth,
            max_num_nodes=max_num_nodes,
            min_leaf_node_size=min_leaf_node_size,
            time_limit=time_limit,
            cost_complexity=0,
            feature_ordering=feature_ordering,
            hyper_tune=hyper_tune,
            use_branch_caching=use_branch_caching,
            use_dataset_caching=False,
            use_terminal_solver=use_terminal_solver,
            use_similarity_lower_bound=use_similarity_lower_bound,
            use_upper_bound=use_upper_bound,
            use_lower_bound=use_lower_bound,
            verbose=verbose,
            random_seed=random_seed,
            continuous_binarize_strategy=continuous_binarize_strategy,
            n_thresholds=n_thresholds,
            n_categories=n_categories)
        self.discrimination_limit = discrimination_limit
        self.sensitive_feature = sensitive_feature
        
    def _initialize_param_handler(self):
        super()._initialize_param_handler()
        self._params.discrimination_limit = self.discrimination_limit
        return self._params

    def _move_sensitive_feature_first(self, X):
        if isinstance(self.sensitive_feature, int):
            if self.sensitive_feature == 0: return X
            if isinstance(X, pd.DataFrame):
                sensitive_feature = X.columns[self.sensitive_feature]
                sensitive_column = X.pop(sensitive_feature) 
                X.insert(0, sensitive_feature, sensitive_column) 
                return X
            return X[:, [self.sensitive_feature, *range(0,self.sensitive_feature), *range(self.sensitive_feature+1, X.shape[1])]]

        elif isinstance(self.sensitive_feature, str):
            # assume pandas dataframe
            if not self.sensitive_feature in X.columns:
                 raise ValueError(f"Column {self.sensitive_feature} is not part of the data.")
            sensitive_column = X.pop(self.sensitive_feature) 
            X.insert(0, self.sensitive_feature, sensitive_column)
            return X
        else:
            raise ValueError(f"The sensitive feature column should be either an index or a string, but is {self.sensitive_feature} of type {type(self.sensitive_feature)}.")

    def fit(self, X, y, categorical=None):
        """
        Fits a STreeD model to the given training data.

        Args:
            x : array-like, shape = (n_samples, n_features)
            Data matrix

            y : array-like, shape = (n_samples)
            Target vector

            categorical : array-like, 
            List of column names that are categorical

        Returns:
            BaseSTreeDSolver

        Raises:
            ValueError: If x or y is None or if they have different number of rows.
        """
        self.n_classes_ = len(np.unique(y))
        X = self._move_sensitive_feature_first(X)
        return super().fit(X, y, categorical)
    
    def predict(self, X):
        """
        Predicts the target variable for the given input feature data.

        Args:
            x : array-like, shape = (n_samples, n_features)
            Data matrix

        Returns:
            numpy.ndarray: A 1D array that represents the predicted target variable of the test data.
                The i-th element in this array corresponds to the predicted target variable for the i-th instance in `x`.
        """
        X = self._move_sensitive_feature_first(X)
        return super().predict(X)
    
    def score(self, X, y_true):
        """
        Computes the score for the given input feature data

        Args:
            x : array-like, shape = (n_samples, n_features)
            Data matrix
            y_true : array-like, shape = (n_samples)
            The true labels

        Returns:
            The score
        """
        X = self._move_sensitive_feature_first(X)
        return super().score(X, y_true)
        
    def _export_dot_leaf_node(self, fh, node, node_id, label_names, train_data):
        if not hasattr(self, "_colors"):
            self._colors = _color_brew(self.n_classes_)
        color = self._colors[node.label]
        return super()._export_dot_leaf_node(fh, node, node_id, label_names, train_data, color=color)