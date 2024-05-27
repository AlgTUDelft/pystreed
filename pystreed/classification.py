from pystreed.base import BaseSTreeDSolver
from typing import Optional
from sklearn.utils._param_validation import Interval
from pystreed.utils import _color_brew
import numpy as np
import numbers
import warnings

class STreeDClassifier(BaseSTreeDSolver):
    """
    STreeDClassifier returns optimal classification trees.
    It supports several objectives, as specified by the optimization task parameter
    """

    def __init__(self, 
                 optimization_task : str = "accuracy",
                 max_depth : int = 3,
                 max_num_nodes : Optional[int] = None,
                 min_leaf_node_size: int = 1,
                 time_limit : float = 600,
                 cost_complexity : float = 0.01,
                 feature_ordering : str = "gini", 
                 hyper_tune: bool = False,
                 use_branch_caching: bool = False,
                 use_dataset_caching: bool = True,
                 use_terminal_solver: bool = True,
                 use_similarity_lower_bound: bool = True,
                 use_upper_bound: bool = True,
                 use_lower_bound: bool = True,
                 upper_bound: float = 2**31 -1,
                 verbose : bool = False,
                 random_seed: int = 27, 
                 continuous_binarize_strategy: str = 'quantile',
                 n_thresholds: int = 5,
                 n_categories: int = 5):
        """
        Construct a STreeDClassifier

        Parameters:
            optimization_task: the objective used for optimization. Default = accuracy
            max_depth: the maximum depth of the tree
            max_num_nodes: the maximum number of branching nodes of the tree
            min_leaf_node_size: the minimum number of training instance that should end up in every leaf node
            time_limit: the time limit in seconds for fitting the tree
            cost_complexity: the cost of adding a branch node, expressed as a percentage. E.g., 0.01 means a branching node may be added if it increases the training accuracy by at least 1%.
                only used when optimization_task == "cost-complex-accuracy'
            feature_ordering: heuristic for the order that features are checked. Default: "gini", alternative: "in-order": the order in the given data
            hyper_tune: Use five-fold validation to tune the size of the tree to prevent overfitting
            use_branch_caching: Enable/Disable branch caching (typically the slower caching strategy. May be faster in some scenario's)
            use_dataset_caching: Enable/Disable dataset caching (typically the faster caching strategy)
            use_terminal_solver: Enable/Disable the depth-two solver (Enabled typically results in a large runtime advantage)
            use_similarity_lower_bound: Enable/Disable the similarity lower bound (Enabled typically results in a large runtime advantage)
            use_upper_bound: Enable/Disable the use of upper bounds (Enabled is typically faster)
            use_lower_bound: Enable/Disable the use of lower bounds (Enabled is typically faster)
            upper_bound: Search for a tree better than the provided upper bound
            verbose: Enable/Disable verbose output
            random_seed: the random seed used by the solver (for example when creating folds)
            continuous_binarization_strategy: the strategy used for binarizing continuous features
            n_thresholds: the number of thresholds to use per continuous feature
            n_categories: the number of categories to use per categorical feature
        """
        if not optimization_task in ["accuracy", "cost-complex-accuracy", "f1-score"]:
            raise ValueError(f"Invalid value for optimization_task: {optimization_task}")
        BaseSTreeDSolver.__init__(self, optimization_task, 
            max_depth=max_depth,
            max_num_nodes=max_num_nodes,
            min_leaf_node_size=min_leaf_node_size,
            time_limit=time_limit,
            cost_complexity=cost_complexity,
            feature_ordering=feature_ordering,
            hyper_tune=hyper_tune,
            use_branch_caching=use_branch_caching,
            use_dataset_caching=use_dataset_caching,
            use_terminal_solver=use_terminal_solver,
            use_similarity_lower_bound=use_similarity_lower_bound,
            use_upper_bound=use_upper_bound,
            use_lower_bound=use_lower_bound,
            upper_bound=upper_bound,
            verbose=verbose,
            random_seed=random_seed,
            continuous_binarize_strategy=continuous_binarize_strategy,
            n_thresholds=n_thresholds,
            n_categories=n_categories)
        if optimization_task == "f1-score" and upper_bound != 2**31-1:
            warnings.warn(f"upper_bound parameter is ignored for f1-score", stacklevel=2)
        
    def _initialize_param_handler(self):
        super()._initialize_param_handler()
        return self._params

    def fit(self, X, y, extra_data=None, categorical=None):
        """
        Fits a STreeD Classification model to the given training data.

        Args:
            x : array-like, shape = (n_samples, n_features)
            Data matrix

            y : array-like, shape = (n_samples)
            Target vector

            extra_data : array-like, shape = (n_samples, n_data_items)
            An array (optional) that represents extra data per instance

            categorical : array-like, 
            List of column names that are categorical

        Returns:
            BaseSTreeDSolver

        Raises:
            ValueError: If x or y is None or if they have different number of rows.
        """
        self.n_classes_ = len(np.unique(y))
        return super().fit(X, y, extra_data, categorical)
        
    def _export_dot_leaf_node(self, fh, node, node_id, label_names, train_data):
        if not hasattr(self, "_colors"):
            self._colors = _color_brew(self.n_classes_)
        color = self._colors[node.label]
        return super()._export_dot_leaf_node(fh, node, node_id, label_names, train_data, color=color)