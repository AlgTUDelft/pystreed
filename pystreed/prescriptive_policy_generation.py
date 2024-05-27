from pystreed.data import PPGData
from pystreed.base import BaseSTreeDSolver
from sklearn.utils._param_validation import Interval, StrOptions
from typing import Optional
from pystreed.utils import _color_brew
import numpy as np

class STreeDPrescriptivePolicyGenerator(BaseSTreeDSolver):

    _parameter_constraints: dict = {**BaseSTreeDSolver._parameter_constraints, 
        "teacher_method": [StrOptions({"DR", "DM", "IPW"})]
    }

    def __init__(self, 
                 max_depth : int = 3,
                 max_num_nodes : Optional[int] = None,
                 min_leaf_node_size : int = 1,
                 teacher_method : str = "DR",
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
                 n_categories: int = 5):
        """
        Construct a STreeDPrescriptivePolicyGenerator

        Parameters:
            max_depth: the maximum depth of the tree
            max_num_nodes: the maximum number of branching nodes of the tree
            min_leaf_node_size: the minimum number of training instance that should end up in every leaf node
            teacher_method: the teacher objective function. Either direct method ("DM"), inverse propensity weighting ("IPW") or doubly robust ("DR")
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
        """
        BaseSTreeDSolver.__init__(self, "prescriptive-policy", 
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
            n_categories=n_categories)
        self.teacher_method = teacher_method
        self.n_classes_ = None

    def _initialize_param_handler(self):
        super()._initialize_param_handler()
        self._params.ppg_teacher_method = self.teacher_method
    
    def _check_data(self, y_in, reset=True):
        y_out = []
        if isinstance(y_in, np.ndarray):
            extra_data = []
            if len(y_in.shape) == 2 and y_in.shape[1] >= 5:
                if reset or self.n_classes_ is None:
                    n_classes = len(np.unique(y_in[:, 0]))
                else:
                    n_classes = self.n_classes_
                for row in y_in:
                    historic_treatment = int(row[0])
                    historic_outcome = float(row[1])
                    propensity_score = float(row[2])
                    if y_in.shape[1] == 4 + n_classes * 2:
                        regress_compare_yhat = [float(row[3 + i]) for i in range(n_classes)]
                        optimal_treatment = int(row[3+n_classes])
                        counterfactual_y = [float(row[4 + n_classes + i]) for i in range(n_classes)]
                        
                        extra_data.append(PPGData(historic_treatment, historic_outcome, 
                                                propensity_score, regress_compare_yhat,
                                                optimal_treatment, counterfactual_y))
                    elif y_in.shape[1] == 3 + n_classes:
                        regress_compare_yhat = [float(row[3 + i]) for i in range(n_classes)]
                        extra_data.append(PPGData(historic_treatment, historic_outcome, 
                                                propensity_score, regress_compare_yhat))
                    else:
                        raise ValueError(f"The array does not have the right shape. Expected shape is either {3+n_classes} columns or {4 + 2*n_classes} columns.")
            else:
                raise ValueError(f"The array does not have the right shape. Expected shape is a two-dimensional array with at least 5 columns.")
        else:
            extra_data = y_in
        
        for i, inst in enumerate(extra_data):
            if not isinstance(inst, PPGData):
                raise ValueError("Each instance in y is expected to be of class PPGData.")
            y_out.append(inst.historic_treatment)

        if reset or self.n_classes_ is None:
            self.n_classes_ = len(np.unique(y_out))
        return np.array(y_out, dtype=np.intc), extra_data

    def _process_extra_data(self, X, extra_data):
        return extra_data

    def fit(self, X, y, categorical=None):
        """
        Fits a STreeD model to the given training data.

        Args:
            x : array-like, shape = (n_samples, n_features)
            Data matrix

            y : array-like with shape (n_samples, n_ppg_columns) 
            Target vector. n_ppg_columns depends on the number of possible treatments K
            * Column 0 is the historic label k (treatment)
            * Column 1 is the historic outcome y 
            * Column 2 is the propensity score mu
            * Column 3 .. 3+K-1 is the regress and compare y_hat prediction
            The rest of the data is optional (for testing)
            * Column 3 + K is the optimal treatment
            * Column 4 + K .. 4 + 2K - 1 is the counterfactual outcome y

            categorical : array-like, 
            List of column names that are categorical

        Returns:
            STreeDPrescriptivePolicyGenerator

        Raises:
            ValueError: If x or y is None or if they have different number of rows.
        """
        y, extra_data = self._check_data(y)
        return super().fit(X, y, extra_data, categorical)
    
    def predict(self, X, extra_data=None):
        """
        Predicts the target variable for the given input feature data.

        Args:
            x : array-like, shape = (n_samples, n_features)
            Data matrix
            extra_data (optional) : array-like with shape (n_samples, n_ppg_columns) 
            Target vector. n_ppg_columns depends on the number of possible treatments K
            * Column 0 is the historic label k (treatment)
            * Column 1 is the historic outcome y 
            * Column 2 is the propensity score mu
            * Column 3 .. 3+K-1 is the regress and compare y_hat prediction
            The rest of the data is optional (for testing)
            * Column 3 + K is the optimal treatment
            * Column 4 + K .. 4 + 2K - 1 is the counterfactual outcome y
            
        Returns:
            numpy.ndarray: A 1D array that represents the predicted target variable of the test data.
                The i-th element in this array corresponds to the predicted target variable for the i-th instance in `x`.
        """
        if extra_data is None:
            extra_data = [PPGData() for i in range(len(X))]
        _, extra_data = self._check_data(extra_data, reset=False)
        return super().predict(X, extra_data)
    
    def score(self, X, y_test):
        """
        Computes the score for the given input feature data

        Args:
            x : array-like, shape = (n_samples, n_features)
            Data matrix
            y_test : array-like with shape (n_samples, n_ppg_columns) 
            Target vector. n_ppg_columns depends on the number of possible treatments K
            * Column 0 is the historic label k (treatment)
            * Column 1 is the historic outcome y 
            * Column 2 is the propensity score mu
            * Column 3 .. 3+K-1 is the regress and compare y_hat prediction
            * Column 3 + K is the optimal treatment
            * Column 4 + K .. 4 + 2K - 1 is the counterfactual outcome y

        Returns:
            The score
        """
        y_test, extra_data = self._check_data(y_test, reset=False)
        return super().score(X, y_test, extra_data)

    def _export_dot_leaf_node(self, fh, node, node_id, label_names, train_data):
        if not hasattr(self, "_colors"):
            self._colors = _color_brew(self.n_classes_)
        color = self._colors[node.label]
        return super()._export_dot_leaf_node(fh, node, node_id, label_names, train_data, color=color)