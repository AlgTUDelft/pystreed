from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from .cstreed import initialize_streed_solver, ParameterHandler
from pystreed.binarizer import Binarizer, _column_threshold
from typing import Optional
import numpy as np
import warnings
import math
import time
import numbers
import sys

class BaseSTreeDSolver(BaseEstimator):

    _parameter_constraints: dict = {
        "optimization_task": [StrOptions({"accuracy", "cost-complex-accuracy", 
                                          "regression", "cost-complex-regression", 
                                          "piecewise-linear-regression", "simple-linear-regression",
                                          "cost-sensitive", "f1-score", "prescriptive-policy", "instance-cost-sensitive",
                                          "group-fairness", "equality-of-opportunity",
                                          "survival-analysis"})],
        "max_depth": [Interval(numbers.Integral, 0, 20, closed="both")],
        "max_num_nodes": [Interval(numbers.Integral, 0, 1048575, closed="both"), None],
        "min_leaf_node_size": [Interval(numbers.Integral, 1, None, closed="left")],
        "time_limit": [Interval(numbers.Real, 0, None, closed="neither")],
        "cost_complexity": [Interval(numbers.Real, 0, 1, closed="both")],
        "feature_ordering": [StrOptions({"in-order", "gini"})],
        "upper_bound": [Interval(numbers.Real, 0, None, closed="left")],
        "random_seed": [Interval(numbers.Integral, -1, None, closed="left")],
        "n_thresholds": [Interval(numbers.Integral, 1, None, closed="left")],
        "n_categories": [Interval(numbers.Integral, 2, None, closed="left")],
        "continuous_binarize_strategy": [StrOptions({"tree", "quantile", "uniform"})],
    }

    def __init__(self, 
            optimization_task: str,
            max_depth: int = 3,
            max_num_nodes: Optional[int] = None,
            min_leaf_node_size: int = 1,
            time_limit: float = 600,
            cost_complexity : float = 0.01,
            feature_ordering : str = "gini", 
            hyper_tune: bool = False,
            use_branch_caching: bool = False,
            use_dataset_caching: bool = True,
            use_terminal_solver: bool = True,
            use_similarity_lower_bound: bool = True,
            use_upper_bound: bool = True,
            use_lower_bound: bool = True,
            upper_bound: float = 2**31-1,
            verbose: bool = False,
            random_seed: int = 27, 
            continuous_binarize_strategy: str = 'quantile',
            n_thresholds: int = 5,
            n_categories: int = 5):
        """
        Construct a BaseSTreeDSolver

        Parameters:
            optimization_task: the objective used for optimization.
            max_depth: the maximum depth of the tree
            max_num_nodes: the maximum number of branching nodes of the tree
            min_leaf_node_size: the minimum number of training instance that should end up in every leaf node
            time_limit: the time limit in seconds for fitting the tree
            cost_complexity: the cost of adding a branch node, expressed as a percentage. E.g., 0.01 means a branching node may be added if it increases the training accuracy by at least 1%.
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
        
        self.optimization_task = optimization_task
        self.max_depth = max_depth
        self.max_num_nodes = max_num_nodes
        self.min_leaf_node_size = min_leaf_node_size
        self.time_limit = time_limit
        self.cost_complexity = cost_complexity
        self.feature_ordering = feature_ordering
        self.hyper_tune = hyper_tune
        self.use_branch_caching = use_branch_caching
        self.use_dataset_caching = use_dataset_caching
        self.use_terminal_solver = use_terminal_solver
        self.use_similarity_lower_bound = use_similarity_lower_bound
        self.use_upper_bound = use_upper_bound
        self.use_lower_bound = use_lower_bound
        self.upper_bound = upper_bound
        self.verbose = verbose
        self.random_seed = random_seed
        self.continuous_binarize_strategy = continuous_binarize_strategy
        self.n_thresholds = n_thresholds
        self.n_categories = n_categories
        
        self.fit_result = None
        self._solver = None
        self._label_type = np.int32
        self.binarizer_ = None
        self._prev_data = None
        self._reset_parameters = ["optimization_task", "cost_complexity", "min_leaf_node_size"]

    def _initialize_param_handler(self):
        max_nodes = int(math.pow(2, self.max_depth) - 1)
        max_num_nodes = self.max_num_nodes
        if max_num_nodes is None:
            max_num_nodes = max_nodes
        else:
            if max_num_nodes > max_nodes:
                warnings.warn(f"Maximum number of branching nodes is reduced to {max_nodes} because of the maximum depth {self.max_depth}.", stacklevel=2)
                max_num_nodes  = max_nodes

        self._params = ParameterHandler()
        self._params.optimization_task = self.optimization_task
        self._params.hyper_tune = self.hyper_tune
        self._params.max_depth = self.max_depth
        self._params.max_num_nodes = max_num_nodes
        self._params.min_leaf_node_size = self.min_leaf_node_size
        self._params.time_limit = self.time_limit
        self._params.cost_complexity = self.cost_complexity
        self._params.feature_ordering = self.feature_ordering
        self._params.verbose = self.verbose
        self._params.random_seed = self.random_seed

        self._params.use_branch_caching = self.use_branch_caching
        self._params.use_dataset_caching = self.use_dataset_caching
        self._params.use_terminal_solver = self.use_terminal_solver
        self._params.use_similarity_lower_bound = self.use_similarity_lower_bound
        self._params.use_upper_bound = self.use_upper_bound
        self._params.use_lower_bound = self.use_lower_bound
        self._params.upper_bound = self.upper_bound
    
    def get_solver_params(self):
        return self._solver._get_parameters()

    def _should_reset_solver(self, X, y, extra_data):
        """
        Check if the solver should reset for a new fit. 
        self._reset_parameters contains a list of parameters that, 
        when changed, invalidate the solver's cache
        """
        if self._solver is None: return True
        if self._prev_data is None: return True
        params = self.get_params()
        changed = []
        for key, val in params.items():
            if not hasattr(self._params, key): continue
            cur_val = getattr(self._params, key)
            if cur_val != val:
                changed.append(key)
        if any(s in changed for s in self._reset_parameters):
            return True
        return False        

    def _process_fit_data(self, X, y=None):
        """
        Validate the X and y data before calling fit
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=FutureWarning)
            X = self._validate_data(X, ensure_min_samples=2, dtype=np.intc)
            
            if not y is None:
                y = check_array(y, ensure_2d=False, dtype=self._label_type)
                if X.shape[0] != y.shape[0]:
                    raise ValueError('x and y have different number of rows')
                return X, y
            return X
    
    def _process_score_data(self, X, y=None):
        """
        Validate the X and y data before calling score
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=FutureWarning)
            X = self._validate_data(X, reset=False, dtype=np.intc)
            
            if not y is None:
                y = check_array(y, ensure_2d=False, dtype=self._label_type)
                if X.shape[0] != y.shape[0]:
                    raise ValueError('x and y have different number of rows')
                return X, y
            return X
    
    def _process_predict_data(self, X):
        """
        Validate the X and y data before calling predict
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=FutureWarning)
            return self._validate_data(X, reset=False, dtype=np.intc)

    def _process_extra_data(self, X, extra_data):
        """
        Process the extra data.
        A subclass should override this function if it uses extra data
        """
        if extra_data is None:
            return []
        raise ValueError("extra data is not empty, but the optimization task does not need extra data.")

    def _binarize_data(self, X, y=None, categorical_columns=None, reset=True):
        """
        Binarize the feature data X. If the binarization strategy is "tree", then the label y is also required.
        Binarize the categorical_columns into at most self.n_categories binary features
        Binarize the non-binary, non-categorical features (the continuous features) into self.n_thresholds binary features
        """
        if reset:
            self.binarizer_ = Binarizer(self.continuous_binarize_strategy, self.n_thresholds, self.n_categories, categorical_columns)
            self.binarizer_.fit(X, y)
        return self.binarizer_.transform(X)

    def _post_initialize_solver(self):
        """
        Override this function in a subclass if the subclass needs to process
        things after the solver is initialized
        """
        pass

    def fit(self, X, y, extra_data=None, categorical=None):
        """
        Fits a STreeD model to the given training data.

        Args:
            x : array-like, shape = (n_samples, n_features)
            Data matrix

            y : array-like, shape = (n_samples)
            Target vector

            extra_data : array-like, shape = (n_samples, n_data_items)
            An array (optional) that represents extra data per instance

        Returns:
            BaseSTreeDSolver

        Raises:
            ValueError: If x or y is None or if they have different number of rows.
        """
        # Validate params and data
        self._validate_params()
        X = self._binarize_data(X, y, categorical)
        X, y = self._process_fit_data(X, y)
        extra_data = self._process_extra_data(X, extra_data)
        # Store train data
        self.train_X_ = X
        self.train_y_ = y
        self.train_extra_data_ = extra_data

        if self._should_reset_solver(X, y, extra_data):
            self._initialize_param_handler()
            self._solver = initialize_streed_solver(self._params)
        else:
            self._initialize_param_handler()
            self._solver._update_parameters(self._params)
        self._post_initialize_solver()
                
        start_time = time.time()
        self.fit_result = self._solver._solve(X, y, extra_data)
        duration = time.time() - start_time
        
        if duration > self.time_limit:
            warnings.warn("Fitting exceeds time limit.", stacklevel=2)
        if not self.fit_result.is_feasible():
            warnings.warn("No feasible tree found.", stacklevel=2)
            delattr(self, "fit_result")
        else:
            self.tree_ = self._solver._get_tree(self.fit_result)

        if self.is_fitted() and self.verbose:
            print("Training score: ", self.fit_result.score())
            print("Tree depth: ", self.fit_result.tree_depth(), " \tBranching nodes: ", self.fit_result.tree_nodes())
            if not self.fit_result.is_optimal():
                print("No proof of optimality!")
        
        return self

    def is_fitted(self):
        return hasattr(self, "fit_result")

    def predict(self, X, extra_data=None):
        """
        Predicts the target variable for the given input feature data.

        Args:
            x : array-like, shape = (n_samples, n_features)
            Data matrix
            extra_data : array-like, shape = (n_samples)
            Extra data (if required)

        Returns:
            numpy.ndarray: A 1D array that represents the predicted target variable of the test data.
                The i-th element in this array corresponds to the predicted target variable for the i-th instance in `x`.
        """
        check_is_fitted(self, "fit_result")
        X = self._binarize_data(X, reset=False)
        X = self._process_predict_data(X)
        extra_data = self._process_extra_data(X, extra_data)
        return self._solver._predict(self.fit_result, X, extra_data)
    
    def score(self, X, y_true, extra_data=None):
        """
        Computes the score for the given input feature data

        Args:
            x : array-like, shape = (n_samples, n_features)
            Data matrix
            y_true : array-like, shape = (n_samples)
            The true labels
            extra_data : array-like, shape = (n_samples)
            Extra data (if required)

        Returns:
            The score
        """
        check_is_fitted(self, "fit_result")
        X = self._binarize_data(X, reset=False)
        X, y_true = self._process_score_data(X, y_true)
        extra_data = self._process_extra_data(X, extra_data)
        self.test_result = self._solver._test_performance(self.fit_result, X, y_true, extra_data)
        return self.test_result.score()

    def get_n_leaves(self):
        """
        Returns the number of branching nodes in the fitted tree
        """
        check_is_fitted(self, "fit_result")
        return self.fit_result.tree_nodes()
    
    def get_depth(self):
        """
        Returns the depth of the fitted tree (a single leaf node is depth zero)
        """
        check_is_fitted(self, "fit_result")
        return self.fit_result.tree_depth()
    
    def get_tree(self):
        """
        Returns the fitted tree
        """
        check_is_fitted(self, "tree_")
        return self.tree_
    
    def _get_label_str(self, label, label_names=None):
        return str(label) if not isinstance(label, int) or label_names is None else label_names[label]
    
    def _get_predicate_str(self, feature, feature_names=None):
        if feature_names is None:
            return f"Feature {feature}"
        feature_name = feature_names[feature]
        if " <= " in feature_name:
            feature_name, threshold = feature_name.split(" <= ")
            if len(threshold) >= 3:
                threshold = _column_threshold(float(threshold))
            return f"{feature_name} {self.__comparator} {threshold}"
        return feature_name

    def _recursive_print_tree(self, out, node, feature_names, label_names, ind=''):
        if node.is_leaf_node():
            label =  self._get_label_str(node.label, label_names)
            out.write(f"{ind}Label: {label}\n")
        else:
            predicate = self._get_predicate_str(node.feature, feature_names)
            out.write(f"{ind}{predicate} not satisfied\n")
            self._recursive_print_tree(out, node.left_child, feature_names, label_names, ind+"|   ")
            out.write(f"{ind}{predicate} satisfied\n")
            self._recursive_print_tree(out, node.right_child, feature_names, label_names, ind+"|   ")

    def print_tree(self, filename=None, feature_names=None, label_names=None):
        """
        Prints the tree on stdout or writes it to a file (if a filename is given)
        If feature_names is not None, use the names in feature_names for pretty printing
        If label_names is not None, use the names in label_names for pretty printing (only for classification)
        """
        check_is_fitted(self, "tree_")
        
        if feature_names is None and hasattr(self, "feature_names_in_"):
            feature_names = self.feature_names_in_
        self.__comparator = "<="

        if filename is None:
            self._recursive_print_tree(sys.stdout, self.tree_, feature_names, label_names)
        else:
            with open(filename, "w") as fh:
                self._recursive_print_tree(fh, self.tree_, feature_names, label_names)

    def export_dot(self, filename, feature_names=None, label_names=None):
        """
        Write a .dot representation of the tree to filename
        If feature_names is not None, use the names in feature_names for pretty printing
        If label_names is not None, use the names in label_names for pretty printing (only for classification)
        """
        check_is_fitted(self, "tree_")

        if feature_names is None and hasattr(self, "feature_names_in_"):
            feature_names = self.feature_names_in_
        train_data = (self.train_X_, self.train_y_, self.train_extra_data_)
        self.__comparator = "&#8804;" # &#8804; is the <= character

        with open(filename, "w", encoding="utf-8") as fh:
            fh.write("digraph Tree {\n")
            fh.write("node [shape=box, style=\"filled, rounded\", fontname=\"helvetica\", fontsize=\"8\"] ;\n")
            fh.write("edge [fontname=\"helvetica\", fontsize=\"6\"] ;\n")
            self._recursive_export_dot(fh, self.tree_, 0, feature_names, label_names, train_data)
            fh.write("}")
    
    def _export_dot_leaf_node(self, fh, node, node_id, label_names, train_data, color=(200, 200, 200)):
        label =  self._get_label_str(node.label, label_names)
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        hex_line_color = "#{:02x}{:02x}{:02x}".format(*[int(0.4 * c) for c in color])
        fh.write(f"{node_id}  [label=\"{label}\", color=\"{hex_line_color}\" fillcolor=\"{hex_color}\"] ;\n")

    def _recursive_export_dot(self, fh, node, node_id, feature_names, label_names, train_data):
        if node.is_leaf_node():
            self._export_dot_leaf_node(fh, node, node_id, label_names, train_data)
        else:
            predicate = self._get_predicate_str(node.feature, feature_names)
            fh.write(f"{node_id} [label=\"{predicate}\", color=\"#222222\", fillcolor=\"#EEEEEE\"] ;\n")
        if node_id > 0:
            parent_id = (node_id - 1) // 2
            feature_label = "True" if (node_id % 2) == 0 else "False"
            angle = 45 if feature_label == "True" else -45
            fh.write(f"{parent_id} -> {node_id} [labeldistance=2.5, labelangle={angle}, label=\"{feature_label}\"] ;\n")

        if node.is_branching_node():
            left_data, right_data = self._split(train_data, node.feature)
            self._recursive_export_dot(fh, node.left_child, node_id * 2 + 1, feature_names, label_names, left_data)
            self._recursive_export_dot(fh, node.right_child, node_id * 2 + 2, feature_names, label_names, right_data)

    def _split(self, data, feature):
        x, y, ed = data
        go_left = x[:, feature] == 0
        left_data =  (x[ go_left], y[ go_left], [e for i, e in enumerate(ed) if go_left[i]])
        right_data = (x[~go_left], y[~go_left], [e for i, e in enumerate(ed) if not go_left[i]])
        return left_data, right_data