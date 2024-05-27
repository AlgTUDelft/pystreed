from pystreed.base import BaseSTreeDSolver
from typing import Optional
from pystreed.utils import _color_brew
import numpy as np
from pystreed.data import CostSpecifier

class STreeDCostSensitiveClassifier(BaseSTreeDSolver):

    def __init__(self, 
                 max_depth : int = 3,
                 max_num_nodes : Optional[int] = None,
                 min_leaf_node_size: int = 1,
                 time_limit : float = 600,
                 feature_ordering : str = "gini", 
                 hyper_tune: bool = False,
                 use_branch_caching: bool = True,
                 use_terminal_solver: bool = True,
                 use_upper_bound: bool = True,
                 use_lower_bound: bool = True,
                 upper_bound: float = 2**31-1,
                 verbose : bool = False,
                 random_seed: int = 27):
        """
        Construct a STreeDCostSensitiveClassifier

        Parameters:
            max_depth: the maximum depth of the tree
            max_num_nodes: the maximum number of branching nodes of the tree
            min_leaf_node_size: the minimum number of training instance that should end up in every leaf node
            time_limit: the time limit in seconds for fitting the tree
            feature_ordering: heuristic for the order that features are checked. Default: "gini", alternative: "in-order": the order in the given data
            hyper_tune: Use five-fold validation to tune the size of the tree to prevent overfitting
            use_branch_caching: Enable/Disable branch caching
            use_terminal_solver: Enable/Disable the depth-two solver (Enabled typically results in a large runtime advantage)
            use_upper_bound: Enable/Disable the use of upper bounds (Enabled is typically faster)
            use_lower_bound: Enable/Disable the use of lower bounds (Enabled is typically faster)
            upper_bound: Search for a tree better than the provided upper bound
            verbose: Enable/Disable verbose output
            random_seed: the random seed used by the solver (for example when creating folds)
        """
        BaseSTreeDSolver.__init__(self, "cost-sensitive", 
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
            use_similarity_lower_bound=False,
            use_upper_bound=use_upper_bound,
            use_lower_bound=use_lower_bound,
            upper_bound=upper_bound,
            verbose=verbose,
            random_seed=random_seed)
        
    def _post_initialize_solver(self):
        super()._post_initialize_solver()
        self._solver.specify_costs(self.cost_specifier_)

    def _binarize_data(self, X, y=None, categorical_columns=None, reset=True):
        return X

    def fit(self, X, y, cost_specifier : CostSpecifier, categorical=None):
        """
        Fits a STreeD model to the given training data.

        Args:
            X : array-like, shape = (n_samples, n_features)
            Data matrix

            y : array-like, shape = (n_samples)
            Target vector

            cost_specifier : CostSpecifier
            An object that describes the misclassification matrix and the feature cost function

            categorical : array-like, 
            List of column names that are categorical

        Returns:
            BaseSTreeDSolver

        Raises:
            ValueError: If x or y is None or if they have different number of rows.
        """
        self.n_classes_ = len(np.unique(y))
        self.cost_specifier_ = cost_specifier
        return super().fit(X, y, categorical)
        
    def _export_dot_leaf_node(self, fh, node, node_id, label_names, train_data):
        if not hasattr(self, "_colors"):
            self._colors = _color_brew(self.n_classes_)
        color = self._colors[node.label]
        return super()._export_dot_leaf_node(fh, node, node_id, label_names, train_data, color=color)