from pystreed.data import CostVector
from pystreed.base import BaseSTreeDSolver
from sklearn.utils._param_validation import Interval, StrOptions
from typing import Optional
from pystreed.utils import _color_brew, _dynamic_float_formatter
import numpy as np
import pandas as pd
import os
from pathlib import Path

class STreeDInstanceCostSensitiveClassifier(BaseSTreeDSolver):

    _parameter_constraints: dict = {**BaseSTreeDSolver._parameter_constraints}

    def __init__(self, 
                 max_depth : int = 3,
                 max_num_nodes : Optional[int] = None,
                 min_leaf_node_size : int = 1,
                 time_limit : float = 600,
                 feature_ordering : str = "gini", 
                 hyper_tune: bool = False,
                 use_branch_caching: bool = False,
                 use_dataset_caching: bool = True,
                 use_terminal_solver: bool = True,
                 use_similarity_lower_bound: bool = True,
                 use_upper_bound: bool = True,
                 use_lower_bound: bool = True,
                 upper_bound: float = 2**31 - 1,
                 verbose : bool = False,
                 random_seed : int = 27,
                 continuous_binarize_strategy: str = 'quantile',
                 n_thresholds: int = 5,
                 n_categories: int = 5,
                 max_num_binary_features: int = None):
        """
        Construct a STreeDInstanceCostSensitiveClassifier

        Parameters:
            max_depth: the maximum depth of the tree
            max_num_nodes: the maximum number of branching nodes of the tree
            min_leaf_node_size: the minimum number of training instance that should end up in every leaf node
            time_limit: the time limit in seconds for fitting the tree
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
            max_num_binary_features: the maximum number of binary features (selected by random forest feature importance)
        """
        BaseSTreeDSolver.__init__(self, "instance-cost-sensitive", 
            max_depth=max_depth,
            max_num_nodes=max_num_nodes,
            min_leaf_node_size=min_leaf_node_size,
            time_limit = time_limit,
            cost_complexity=0,
            feature_ordering=feature_ordering,
            hyper_tune = hyper_tune,
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
            n_categories=n_categories,
            max_num_binary_features=max_num_binary_features)
        self.n_classes_ = None

    def _initialize_param_handler(self):
        super()._initialize_param_handler()
    
    def _check_data(self, y_in, reset=True):
        y_out = []
        if isinstance(y_in, pd.DataFrame):
            y_in = y_in.values
        if isinstance(y_in, np.ndarray):
            extra_data = []
            if len(y_in.shape) != 2 or y_in.shape[1] < 2:
                raise ValueError(f"The y-array does not have the right shape. Expected shape is a two-dimensional array with at least 2 columns.")
            if reset or self.n_classes_ is None:
                n_classes = y_in.shape[1]
            else:
                n_classes = self.n_classes_
                if n_classes != y_in.shape[1]:
                    raise ValueError(f"The y-array does not have the expected shape. Expected {n_classes} columns, but found {y_in.shape[1]}.")
            for row in y_in:
                extra_data.append(CostVector(row))
        else:
            extra_data = y_in
        
        num_labels = 0
        for i, inst in enumerate(extra_data):
            if not isinstance(inst, CostVector):
                raise ValueError("Each instance in y is expected to be of class CostVector.")
            y_out.append(np.argmin(inst.costs))
            num_labels = max(num_labels, len(inst.costs))

        if reset or self.n_classes_ is None:
            self.n_classes_ = num_labels
        return np.array(y_out, dtype=np.intc), extra_data

    def _process_extra_data(self, X, extra_data):
        return extra_data

    def fit(self, X, y, categorical=None):
        """
        Fits a STreeD model to the given training data.

        Args:
            x : array-like, shape = (n_samples, n_features)
            Data matrix

            y : array-like with shape (n_samples, n_labels) 
            Cost vector. For each instance specify the classification costs per label

            categorical : array-like, 
            List of column names that are categorical

        Returns:
            STreeDInstanceCostSensitiveClassifier

        Raises:
            ValueError: If x or y is None or if they have different number of rows.
        """
        y, extra_data = self._check_data(y)
        return super().fit(X, y, extra_data)
    
    def predict(self, X, extra_data=None):
        """
        Predicts the target variable for the given input feature data.

        Args:
            x : array-like, shape = (n_samples, n_features)
            Data matrix
            extra_data (optional) :array-like with shape (n_samples, n_labels) 
            Cost vector. For each instance specify the classification costs per label
            
        Returns:
            numpy.ndarray: A 1D array that represents the predicted target variable of the test data.
                The i-th element in this array corresponds to the predicted target variable for the i-th instance in `x`.
        """
        if extra_data is None:
            extra_data = [CostVector(np.zeros(self.n_classes_)) for i in range(len(X))]
        _, extra_data = self._check_data(extra_data, reset=False)
        return super().predict(X, extra_data)
    
    def score(self, X, y_test):
        """
        Computes the score for the given input feature data

        Args:
            x : array-like, shape = (n_samples, n_features)
            Data matrix
            y_test : array-like with shape (n_samples, n_labels) 
            Cost vector. For each instance specify the classification costs per label

        Returns:
            The score
        """
        y_test, extra_data = self._check_data(y_test, reset=False)
        return super().score(X, y_test, extra_data)

    def export_dot(self, filename, feature_names=None, label_names=None):
        """
        Export the tree as a .dot file (for plotting)
        """
        self.__export_directory = Path(filename).parent.absolute() / "leaf-distributions"
        return super().export_dot(filename, feature_names, label_names)
    
    def _get_branching_color(self, cost_sums):
        if not hasattr(self, "colors"):
            self.colors = _color_brew(self.n_classes_)
        best_ix = int(np.argmin(cost_sums))
        color = list(self.colors[best_ix])
        total_costs = sum(cost_sums)
        if cost_sums[best_ix] < 0.01 * total_costs:
            alpha = 0.0
        else:
            sorted_costs = sorted(cost_sums)
            if sorted_costs[1] < 0.01:
                alpha = 0.0
            else:
                alpha = (sorted_costs[1] - sorted_costs[0]) / (sorted_costs[1])
        color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color]
        # Return html color code in #RRGGBB format
        return "#%2x%2x%2x" % tuple(color)

    def _export_bar_plot(self, fh, node_id, label_names, train_data, text_prefix=""):
        if not hasattr(self, "_colors"):
            self._colors = _color_brew(self.n_classes_)
        import matplotlib.pyplot as plt
        plt.rc('axes', labelsize=6)
        plt.rc('xtick', labelsize=6)
        plt.rc('ytick', labelsize=6)
        plt.rc("axes", titlesize=7)
        cost_sums = np.zeros(self.n_classes_)
        costs = train_data[2]
        n = len(costs)
        for row in costs:
            for label in range(self.n_classes_):
                cost_sums[label] += row.costs[label]
        cost_sums /= n
        best_label = int(np.argmin(cost_sums))
        color = self._colors[best_label]

        colors = ['tab:gray' for label in range(self.n_classes_)]
        colors[best_label] = [c / 256 for c in color]
        _label_names = [self._get_label_str(i, label_names) for i in range(self.n_classes_)]
        label_costs = zip(_label_names, cost_sums, colors)
        label_costs = sorted(label_costs, key=lambda i: i[1])[:4]
        _label_names, cost_sums, colors = [list(i) for i in list(zip(*label_costs))]
        _label_names.reverse()
        cost_sums.reverse()
        colors.reverse()

        plt.figure(figsize=(1.8,1.2))
        plt.barh(_label_names, cost_sums, color=colors)
        label =  self._get_label_str(best_label, label_names)
        plt.title(f"{text_prefix} N = {n}, Best = {label}, Costs = {_dynamic_float_formatter(min(cost_sums))}")

        filename = self.__export_directory / f"leaf_ics_{node_id}.png"
        if not os.path.exists(self.__export_directory):
            os.makedirs(self.__export_directory)
        plt.savefig(filename, bbox_inches="tight", dpi=600)
        
        fh.write(f"{node_id}  [label=\"\", image=\"{filename}\", width=2.0, height=1.3, fixedsize=true, shape=none] ;\n")
    
    def _export_dot_predicate_node(self, fh, node, node_id, feature_names, label_names, train_data):
        predicate = self._get_predicate_str(node.feature, feature_names)
        try:
            self._export_bar_plot(fh, node_id, label_names, train_data, f"{predicate}\n")
        except:
            cost_sums = np.zeros(self.n_classes_)
            costs = train_data[2]
            n = len(costs)
            for row in costs:
                for label in range(self.n_classes_):
                    cost_sums[label] += row.costs[label]
            cost_sums /= n
            best_ix = int(np.argmin(cost_sums))
            best_label = self._get_label_str(best_ix, label_names)
            best_costs = cost_sums[best_ix]
            branching_color = self._get_branching_color(cost_sums)
            fh.write(f"{node_id} [label=\"{predicate}\nN = {n}, Best = {best_label}, Costs = {_dynamic_float_formatter(best_costs)}\", color=\"#222222\", fillcolor=\"{branching_color}\"] ;\n") 

    def _export_dot_leaf_node(self, fh, node, node_id, label_names, train_data):
        try:
            self._export_bar_plot(fh, node_id, label_names, train_data)
        except:
            if not hasattr(self, "_colors"):
                self._colors = _color_brew(self.n_classes_)
            color = self._colors[node.label]
            label =  self._get_label_str(node.label, label_names)
            hex_color = "#{:02x}{:02x}{:02x}".format(*color)
            hex_line_color = "#{:02x}{:02x}{:02x}".format(*[int(0.4 * c) for c in color])
            fh.write(f"{node_id}  [label=\"{label}\", color=\"{hex_line_color}\" fillcolor=\"{hex_color}\"] ;\n")