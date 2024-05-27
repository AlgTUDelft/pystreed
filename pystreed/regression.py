from pystreed.base import BaseSTreeDSolver
from pystreed.data import ContinuousFeatureData, SimpleContinuousFeatureData
from pystreed.binarizer import get_column_types
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._param_validation import Interval, StrOptions
from typing import Optional
import numpy as np
import pandas as pd
import numbers
from pathlib import Path
import os

class STreeDRegressor(BaseSTreeDSolver):

    _parameter_constraints: dict = {**BaseSTreeDSolver._parameter_constraints, 
        "regression_lower_bound": [StrOptions({"equivalent", "kmeans"})]
    }

    def __init__(self,
                 optimization_task : str = "cost-complex-regression",
                 max_depth : int = 3,
                 max_num_nodes : Optional[int] = None,
                 min_leaf_node_size : int = 1,
                 time_limit : float = 600,
                 hyper_tune : bool = False,
                 cost_complexity : float = 0.01,
                 regression_lower_bound : str = "kmeans",
                 use_branch_caching: bool = True,
                 use_dataset_caching: bool = False,
                 use_terminal_solver: bool = True,
                 use_similarity_lower_bound: bool = True,
                 use_upper_bound: bool = True,
                 use_lower_bound: bool = True,
                 use_task_lower_bound: bool = True,
                 upper_bound: float = 2**31 - 1,
                 verbose : bool = False,
                 random_seed: int = 27, 
                 continuous_binarize_strategy: str = 'quantile',
                 n_thresholds: int = 5,
                 n_categories: int = 5):
        """
        Construct a STreedRegressor

        Parameters:
            optimization_task: the objective used for optimization. Default = cost-complex-regression
            max_depth: the maximum depth of the tree
            max_num_nodes: the maximum number of branching nodes of the tree
            min_leaf_node_size: the minimum number of training instance that should end up in every leaf node
            time_limit: the time limit in seconds for fitting the tree
            hyper_tune: Use five-fold validation to tune the size of the tree to prevent overfitting
            cost_complexity: the cost of adding a branch node, expressed as a percentage. E.g., 0.01 means a branching node may be added if it increases the training accuracy by at least 1%.
                only used when optimization_task == "cost-complex-regression'
            regression_lower_bound: the lower bound used by the cost-complex-regression task: kmeans or equivalent
            use_branch_caching: Enable/Disable branch caching (typically the slower caching strategy. May be faster in some scenario's)
            use_dataset_caching: Enable/Disable dataset caching (typically the faster caching strategy)
            use_terminal_solver: Enable/Disable the depth-two solver (Enabled typically results in a large runtime advantage)
            use_similarity_lower_bound: Enable/Disable the similarity lower bound (Enabled typically results in a large runtime advantage)
            use_upper_bound: Enable/Disable the use of upper bounds (Enabled is typically faster)
            use_lower_bound: Enable/Disable the use of lower bounds (Enabled is typically faster)
            use_task_lower_bound: Enable/Disable the kmeans/equivalent lower bound for cost-complex-regression (typically faster)
            upper_bound: Search for a tree better than the provided upper bound
            verbose: Enable/Disable verbose output
            random_seed: the random seed used by the solver (for example when creating folds)
            continuous_binarization_strategy: the strategy used for binarizing continuous features
            n_thresholds: the number of thresholds to use per continuous feature
            n_categories: the number of categories to use per categorical feature
        """
        self._extra_permitted_params = ["regression_lower_bound"]
        if not optimization_task in ["regression", "cost-complex-regression"]:
            raise ValueError(f"Invalid value for optimization_task: {optimization_task}")
        BaseSTreeDSolver.__init__(self, "regression",
            max_depth=max_depth,
            max_num_nodes=max_num_nodes,
            min_leaf_node_size=min_leaf_node_size,
            time_limit=time_limit,
            cost_complexity=cost_complexity,
            feature_ordering="in-order",
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
        self._label_type = np.double
        self.regression_lower_bound = regression_lower_bound
        self.use_task_lower_bound = use_task_lower_bound

    def _initialize_param_handler(self):
        super()._initialize_param_handler()
        self._params.regression_lower_bound = self.regression_lower_bound
        self._params.use_task_lower_bound = self.use_task_lower_bound
        return self._params

class STreeDPiecewiseLinearRegressor(BaseSTreeDSolver):

    _parameter_constraints: dict = {**BaseSTreeDSolver._parameter_constraints, 
        "lasso_penalty": [Interval(numbers.Real, 0, 1e12, closed="both")],
        "ridge_penalty": [Interval(numbers.Real, 0, 1e12, closed="both")],
    }

    def __init__(self,
                 simple : bool = False,
                 max_depth : int = 3,
                 max_num_nodes : Optional[int] = None,
                 min_leaf_node_size : int = 1,
                 time_limit : float = 600,
                 hyper_tune : bool = False,
                 cost_complexity : float = 0.01,
                 lasso_penalty : float = 0.0,
                 ridge_penalty : float = 0.0,
                 use_branch_caching: bool = False,
                 use_dataset_caching: bool = True,
                 use_similarity_lower_bound: bool = True,
                 use_upper_bound: bool = True,
                 use_lower_bound: bool = True,
                 upper_bound: float = 2**31 - 1,
                 verbose : bool = False,
                 random_seed: int = 27, 
                 continuous_binarize_strategy: str = 'quantile',
                 n_thresholds: int = 5,
                 n_categories: int = 5):
        """
        Construct a STreeDPiecewiseLinearRegressor

        Parameters:
            simple: set to True to fit a simple linear regression model. False otherwise
            max_depth: the maximum depth of the tree
            max_num_nodes: the maximum number of branching nodes of the tree
            min_leaf_node_size: the minimum number of training instance that should end up in every leaf node
            time_limit: the time limit in seconds for fitting the tree
            hyper_tune: Use five-fold validation to tune the size of the tree to prevent overfitting
            cost_complexity: the cost of adding a branch node, expressed as a percentage. E.g., 0.01 means a branching node may be added if it increases the training accuracy by at least 1%.
            lasso_penalty: the amount of lasso penalization used in each leaf node
            ridge_penalty: the amount of ridge penalization in each leaf node
            use_branch_caching: Enable/Disable branch caching (typically the slower caching strategy. May be faster in some scenario's)
            use_dataset_caching: Enable/Disable dataset caching (typically the faster caching strategy)
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
        BaseSTreeDSolver.__init__(self, "simple-linear-regression" if simple else "piecewise-linear-regression",
            max_depth=max_depth,
            max_num_nodes=max_num_nodes,
            min_leaf_node_size=min_leaf_node_size,
            time_limit=time_limit,
            cost_complexity=cost_complexity,
            feature_ordering="in-order",
            hyper_tune = hyper_tune,
            use_branch_caching=use_branch_caching,
            use_dataset_caching=use_dataset_caching,
            use_terminal_solver=False,
            use_similarity_lower_bound=use_similarity_lower_bound,
            use_upper_bound=use_upper_bound,
            use_lower_bound=use_lower_bound,
            upper_bound=upper_bound,
            verbose=verbose,
            random_seed=random_seed,
            continuous_binarize_strategy=continuous_binarize_strategy,
            n_thresholds=n_thresholds,
            n_categories=n_categories)
        self.simple = simple
        self._label_type = np.double
        self.label_name_ = None
        self.continuous_columns_ = None
        self.n_continuous_columns_ = 0
        self.lasso_penalty = lasso_penalty
        self.ridge_penalty = ridge_penalty
        self._reset_parameters.append("lasso_penalty")
        self._reset_parameters.append("ridge_penalty")

    def _initialize_param_handler(self):
        super()._initialize_param_handler()
        self._params.lasso_penalty = self.lasso_penalty
        self._params.ridge_penalty = self.ridge_penalty
        n_vars = 1 if self.simple else self.n_continuous_columns_
        if self.min_leaf_node_size < 5 * n_vars:
            if self.verbose:
                print(f"Updating the minimum leaf node size to {5 * n_vars}.")
            self._params.min_leaf_node_size = 5 * n_vars
        return self._params

    def _process_extra_data(self, X, extra_data):
        return extra_data

    def _get_continuous_columns(self, X, continuous_columns):
        if continuous_columns is None:
            continuous_columns = self.continuous_columns_
        if continuous_columns is None:
            _, _, continuous_columns = get_column_types(X, self.categorical_columns_)
            if len(continuous_columns) == 0:
                raise ValueError("Provide continuous columns.")
        if isinstance(continuous_columns, list): # column names
            self.continuous_columns_ = continuous_columns
            if isinstance(X, np.ndarray):
                cont_cols = X[:, continuous_columns]
            elif isinstance(X, pd.DataFrame):
                cont_cols = X[continuous_columns].values
            else:
                raise ValueError(f"X should be a numpy array or a pandas data frame. X.type = {type(X)}")
        elif isinstance(continuous_columns, np.ndarray):
            cont_cols = continuous_columns
        elif isinstance(continuous_columns, pd.DataFrame):
            cont_cols = continuous_columns.values
        else:
            raise ValueError(f"X should be a numpy array or a pandas data frame. X.type = {type(X)}")
        self.n_continuous_columns_ = cont_cols.shape[1]
        if self.simple:
            extra_data = [SimpleContinuousFeatureData(list(row)) for row in cont_cols]
        else:
            extra_data = [ContinuousFeatureData(list(row)) for row in cont_cols]
        return extra_data

    def fit(self, X, y, continuous_columns=None, categorical_columns=None):
        """
        Fits a STreeD Regression model to the given training data.

        Args:
            x : array-like, shape = (n_samples, n_features)
            Data matrix

            y : array-like, shape = (n_samples)
            Target vector

            extra_data : array-like, shape = (n_samples, n_data_items)
            An array (optional) that represents extra data per instance

            continuous_columns : array-like
            List of columns that are continuous (will be used for the regression)
            If None, continuous columns are automatically detected

            categorical_columns : array-like, 
            List of column names that are categorical

        Returns:
            BaseSTreeDSolver

        Raises:
            ValueError: If x or y is None or if they have different number of rows.
        """
        self.categorical_columns_ = categorical_columns
        extra_data = self._get_continuous_columns(X, continuous_columns)
        if isinstance(y, pd.Series): self.label_name_ = y.name
        return super().fit(X, y, extra_data, categorical_columns)

    def predict(self, X, continuous_columns=None):
        check_is_fitted(self, "fit_result")
        extra_data = self._get_continuous_columns(X, continuous_columns)
        return super().predict(X, extra_data)
    
    def score(self, X, y_true, continuous_columns=None):
        check_is_fitted(self, "fit_result")
        extra_data = self._get_continuous_columns(X, continuous_columns)
        return super().score(X, y_true, extra_data)
    
    def _export_dot_leaf_node(self, fh, node, node_id, label_names, train_data):
        try:
            x, y, ed = train_data
            import matplotlib.pyplot as plt
            plt.rc('axes', labelsize=6)
            plt.rc('xtick', labelsize=6)
            plt.rc('ytick', labelsize=6)
            linear_model = node.label
            # plot on the axis with the highest difference in predicted values
            nCF = len(linear_model.coefficients)
            x_index = 0
            max_diff = 0
            for index in range(nCF):
                xj = [e.feature_data[index] for e in ed]
                yhat = [_x * linear_model.coefficients[index] for _x in xj]
                diff = max(yhat) - min(yhat)
                if diff > max_diff:
                    x_index = index
                    max_diff = diff
            intercept = linear_model.intercept
            coef = linear_model.coefficients[x_index]
            # add the mean for other features multiplied by the coefficient to the intercept
            for index in range(nCF):
                if index == x_index: continue
                xj = [e.feature_data[index] for e in ed]
                intercept += linear_model.coefficients[index] * np.mean(xj)
            xj = [e.feature_data[x_index] for e in ed]
            yhat = [intercept + _x * coef for _x in xj]
            xj, yhat, y = list(zip(*sorted(zip(xj, yhat, y), key=lambda v: v[0])))
            # plot the figure 
            plt.figure(figsize=(2,2))            
            plt.grid(zorder=-2.0)
            plt.plot(xj, yhat, zorder=3)
            plt.scatter(xj, y, color='red', s=2, zorder=2)
            
            if not label_names is None and isinstance(label_names, str): plt.ylabel()
            elif not self.label_name_ is None: plt.ylabel("(Predicted) Value" if self.label_name_==0 else self.label_name_)
            if isinstance(self.continuous_columns_[x_index], int):
                plt.xlabel(f"Feature {self.continuous_columns_[x_index]}")  
            else:
                plt.xlabel(f"{self.continuous_columns_[x_index]}")
            filename = self.__export_directory / f"leaf_reg_{node_id}.png"
            if not os.path.exists(self.__export_directory):
                os.makedirs(self.__export_directory)
            plt.savefig(filename, bbox_inches="tight", dpi=600)
            fh.write(f"{node_id}  [label=\"\", image=\"{filename}\", width=2, height=2, fixedsize=true, shape=none] ;\n")
        except:
            return super()._export_dot_leaf_node(fh, node, node_id, label_names, train_data)

    def export_dot(self, filename, feature_names=None, label_names=None):
        """
        Export the tree as a .dot file (for plotting)
        """
        self.__export_directory = Path(filename).parent.absolute() / "leaf-distributions"
        return super().export_dot(filename, feature_names, label_names)
    
