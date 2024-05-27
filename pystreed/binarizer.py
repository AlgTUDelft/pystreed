import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _check_feature_names_in, check_array, check_is_fitted
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import sklearn
import numbers
import itertools
import math
import warnings
import collections

def get_column_types(X, categorical_columns=None):
    if categorical_columns is None: 
        categorical_columns = []
    binary_columns = []
    if isinstance(X, pd.DataFrame):
        categorical_columns = categorical_columns + [col for col in X.columns if not col in categorical_columns and not is_numeric_dtype(X[col])]
        binary_columns = [col for col in X.columns if not set(X[col].unique()) - set([0,1])] 
        continuous_columns = [col for col in X.columns if not col in categorical_columns and not col in binary_columns]
    elif isinstance(X, np.ndarray):
        binary_columns = [col for col in range(X.shape[1]) if not set(np.unique(X[:, col])) - set([0,1])]
        continuous_columns = [col for col in range(X.shape[1]) if not col in categorical_columns and not col in binary_columns]
    else:
        raise ValueError("X should be either a numpy array or a pandas data frame.")
    return binary_columns, categorical_columns, continuous_columns

def _escape(obj):
    if isinstance(obj, list):
        return [_escape(j) for j in obj]
    elif isinstance(obj, str):
        return obj.replace("'", "\\'")
    else:
        return obj

class Binarizer:

    def __init__(self, continuous_strategy, n_thresholds, n_categories, categorical_columns):
        """
        Initialize a binarizer

        Parameters:
        continuous_strategy: str, the binarization strategy to use for continuous features. Either "quantile", "uniform" or "tree"
        n_thresholds: int or array-like, the number of tresholds to use for continuous features. If an array, it should have a number for every continuous feature
        n_categories: int or array-like, the number of categories to turn every categorical feature into. If an array, it should have a number for every categorical feature
        categorical_columns: array-like, the names of the categorical feature columns
        """
        self.continuous_strategy = continuous_strategy
        self.n_thresholds = n_thresholds
        self.n_categories = n_categories
        self.categorical_columns = [] if categorical_columns is None else categorical_columns
        self.binary_columns = []
        self.continuous_columns = []
        self.categorical_binarizer = None
        self.continuous_binarizer = None

    def fit(self, X, y):
        X_cat = None
        X_bin = None
        X_cont = None
        self.binary_columns, self.categorical_columns, self.continuous_columns = get_column_types(X, self.categorical_columns)
        if isinstance(X, pd.DataFrame):
            if len(self.categorical_columns) > 0:
                X_cat = X[self.categorical_columns]
            if len(self.binary_columns) > 0:
                X_bin = X[self.binary_columns]
            if len(self.continuous_columns) > 0:
                X_cont = X[self.continuous_columns]
        else:
            if len(self.categorical_columns) > 0:
                X_cat = X[:, self.categorical_columns]
            if len(self.binary_columns) > 0:
                X_bin = X[:, self.binary_columns]
            if len(self.continuous_columns) > 0:
                X_cont = X[:, self.continuous_columns]

        if len(self.categorical_columns) > 0:
            self.categorical_binarizer = CategoricalBinarizer(self.n_categories) 
            self.categorical_binarizer.fit(X_cat)
        if len(self.continuous_columns) > 0:
            self.continuous_binarizer = KThresholdBinarizer(self.continuous_strategy, self.n_thresholds)
            self.continuous_binarizer.fit(X_cont, y)

    def transform(self, X):
        parts = []

        if isinstance(X, pd.DataFrame):
            if len(self.categorical_columns) > 0:
                X_cat = X[self.categorical_columns]
            if len(self.binary_columns) > 0:
                X_bin = X[self.binary_columns]
            if len(self.continuous_columns) > 0:
                X_cont = X[self.continuous_columns]
        else:
            if len(self.categorical_columns) > 0:
                X_cat = X[:, self.categorical_columns]
            if len(self.binary_columns) > 0:
                X_bin = X[:, self.binary_columns]
            if len(self.continuous_columns) > 0:
                X_cont = X[:, self.continuous_columns]
        if len(self.binary_columns) > 0:
            if isinstance(X_bin, pd.DataFrame) and not is_numeric_dtype(X_bin):
                X_bin.columns = [f"Binary Feature {i+1}" for i in range(X_bin.shape[1])] 
                X = X.rename(str,axis="columns")
            parts.append(X_bin)
        if len(self.categorical_columns) > 0:
            X_cat = self.categorical_binarizer.transform(X_cat)
            parts.append(X_cat)
        if len(self.continuous_columns) > 0:
            X_cont = self.continuous_binarizer.transform(X_cont)
            parts.append(X_cont)
                
        if isinstance(X, pd.DataFrame):
            for p in parts:
                p.index = X.index
            return pd.concat(parts, axis=1)
        return np.concatenate(parts, axis=1)

class CategoricalBinarizer(BaseEstimator, TransformerMixin):

    _parameter_constraints: dict = {
        "n_categories": [Interval(numbers.Integral, 1, None, closed="left"), "array-like"],
        "minimum_coverage": [Interval(numbers.Real, 0, 1, closed="both")]
    }
    
    def __init__(self, n_categories=10, minimum_coverage=0.01):
        self.n_categories = n_categories
        self.minimum_coverage = minimum_coverage

    def _validate_n_categories(self, n_features):
        """Returns n_categories_, the number of categories per feature."""
        orig_categories = self.n_categories
        if isinstance(orig_categories, numbers.Integral):
            return np.full(n_features, orig_categories, dtype=int)

        n_categories = check_array(orig_categories, dtype=int, copy=True, ensure_2d=False)

        if n_categories.ndim > 1 or n_categories.shape[0] != n_features:
            raise ValueError("n_categories must be a scalar or array of shape (n_features,).")

        bad_n_categories_value = (n_categories < 1) | (n_categories != orig_categories)

        violating_indices = np.where(bad_n_categories_value)[0]
        if violating_indices.shape[0] > 0:
            indices = ", ".join(str(i) for i in violating_indices)
            raise ValueError(
                "{} received an invalid number "
                "of categories at indices {}. Number of categories "
                "must be at least 1, and must be an int.".format(
                    CategoricalBinarizer.__name__, indices
                )
            )
        return n_categories
    
    def fit(self, X, y=None):
        self._validate_params()
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=FutureWarning)
            X = self._validate_data(X, dtype=None)

        n_samples, n_features = X.shape
        n_categories = self._validate_n_categories(n_features)
        categories = []
        column_names = []
        for jj in range(n_features):
            column = X[:, jj]
            if hasattr(self, "feature_names_in_") and len(self.feature_names_in_) == n_features:
                column_name = self.feature_names_in_[jj]
            else:
                column_name = f"Feature {jj + 1}"
            column_dtype = column.dtype
            counter = collections.Counter(column)
            
            if n_categories[jj] <= 1 or len(counter) <= 1:
                n_categories[jj] = 0
                categories.append([])
                column_names.append([])
                continue
            elif n_categories[jj] == 2 or len(counter) == 2:
                value = counter.most_common(2)[1][0] # get the key of the second most common value
                n_categories[jj] = 1
                categories.append([value])
                escaped_value = _escape(value)
                if isinstance(escaped_value, str):
                    column_names.append([f"{column_name} == '{value}'"])
                else:
                    column_names.append([f"{column_name} == {value}"])
                continue
            values = list(counter.keys())
            if len(values) > n_categories[jj]:
                values = values[:n_categories[jj] - 1] + [values[n_categories[jj] - 1:]]
            n_categories[jj] = len(values)
            categories.append(values)
            new_column_names = []
            escape_values = _escape(values)
            for v in escape_values:
                if isinstance(v, list):
                    new_column_names.append("{} in [{}]".format(column_name, \
                        ",".join([f"'{o}'" if isinstance(o, str) else str(o) for o in v])))
                elif isinstance(v, str):
                    new_column_names.append("{} == '{}'".format(column_name, v))
                else:
                    new_column_names.append("{} == {}".format(column_name, v))
            column_names.append(new_column_names)

        self.n_categories_ = n_categories
        self.categories_ = categories
        self.column_names_ = column_names

    def transform(self, X, y=None):
        check_is_fitted(self)
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=FutureWarning)
            X = self._validate_data(X, reset=False, dtype=None)

        n_samples, n_features = X.shape
        sum_of_bin_features = sum(self.n_categories_)
        bin_X = np.zeros((n_samples, sum_of_bin_features), dtype=np.intc)
        col_ix = 0
        for jj in range(n_features):
            if self.n_categories_[jj] == 0: continue
            column = X[:, jj]
            for c_ix in range(self.n_categories_[jj]):
                category_val = self.categories_[jj][c_ix]
                if isinstance(category_val, list):
                    bin_X[:, col_ix] = np.array([v in category_val for v in column])
                else:
                    bin_X[:, col_ix] = column == category_val
                col_ix += 1
        column_names = itertools.chain.from_iterable(self.column_names_)
        return pd.DataFrame(bin_X, columns=column_names)

class KThresholdBinarizer(BaseEstimator, TransformerMixin):
    
    _parameter_constraints: dict = {
        "strategy": [StrOptions({"tree", "quantile", "uniform"})],
        "n_thresholds": [Interval(numbers.Integral, 1, None, closed="left"), "array-like"]
    }

    def __init__(self, strategy="quantile", n_thresholds=10):
        self.strategy = strategy
        self.n_thresholds = n_thresholds

    def _validate_n_thresholds(self, n_features):
        """Returns n_thresholds_, the number of thresholds per feature."""
        orig_thresholds = self.n_thresholds
        if isinstance(orig_thresholds, numbers.Integral):
            return np.full(n_features, orig_thresholds, dtype=int)

        n_thresholds = check_array(orig_thresholds, dtype=int, copy=True, ensure_2d=False)

        if n_thresholds.ndim > 1 or n_thresholds.shape[0] != n_features:
            raise ValueError("n_thresholds must be a scalar or array of shape (n_features,).")

        bad_nthresholds_value = (n_thresholds < 1) | (n_thresholds != orig_thresholds)

        violating_indices = np.where(bad_nthresholds_value)[0]
        if violating_indices.shape[0] > 0:
            indices = ", ".join(str(i) for i in violating_indices)
            raise ValueError(
                "{} received an invalid number "
                "of thresholds at indices {}. Number of thresholds "
                "must be at least 1, and must be an int.".format(
                    KThresholdBinarizer.__name__, indices
                )
            )
        return n_thresholds

    def fit(self, X, y=None):
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=FutureWarning)
            X = self._validate_data(X, dtype="numeric")
            
        if self.strategy == 'tree':
            if y is None:
                raise ValueError("Binarizing with the tree strategy requires the labels y.")
            with warnings.catch_warnings():
                warnings.filterwarnings(action="ignore", category=FutureWarning)
                y = check_array(y, ensure_2d=False)
            if X.shape[0] != y.shape[0]:
                raise ValueError("Number of samples in X and y are not the same.")

        n_samples, n_features = X.shape
        n_thresholds = self._validate_n_thresholds(n_features)
        
        thresholds = []
        column_names = []
        for jj in range(n_features):
            column = X[:, jj]
            if hasattr(self, "feature_names_in_") and len(self.feature_names_in_) == n_features:
                column_name = self.feature_names_in_[jj]
            else:
                column_name = f"Feature {jj + 1}"
            column_dtype = column.dtype

            col_min, col_max = column.min(), column.max()
            if n_thresholds[jj] == 0 or col_min == col_max:
                n_thresholds[jj] = 0
                thresholds.append([])
                column_names.append([])
                continue
            if self.strategy in ["quantile", "uniform"]:
                if len(np.unique(column)) <=  n_thresholds[jj] + 1:
                    uniques = np.unique(column)
                    cutpoints = [(uniques[i] + uniques[i+1]) / 2 for i in range(len(uniques) - 1)]
                elif self.strategy == "uniform":
                    uniques = np.linspace(col_min, col_max, n_thresholds[jj] + 1)
                    cutpoints = np.unique([(uniques[i] + uniques[i+1]) / 2 for i in range(len(uniques) - 1)])
                else:
                    quantiles = np.linspace(0, 1, n_thresholds[jj] + 2)[1:-1]
                    cutpoints = np.unique(np.quantile(column, quantiles))
            elif self.strategy == 'tree':
                _X = np.array(column).reshape(-1, 1)
                is_classification = issubclass(y.dtype.type, np.integer) and len(np.unique(y)) <= 10
                Ml_model = DecisionTreeClassifier if is_classification else DecisionTreeRegressor
                #error_function = 'accuracy' if is_classification else "neg_mean_squared_error"
                model = Ml_model(max_leaf_nodes=n_thresholds[jj] + 1)
                model.fit(_X, y)
                cutpoints = sorted([t for t in model.tree_.threshold if t != sklearn.tree._tree.TREE_UNDEFINED])
            n_thresholds[jj] = len(cutpoints)
            thresholds.append(cutpoints)
            column_names.append([f"{column_name} <= {c}" for c in cutpoints])
        
        self.n_thresholds_ = n_thresholds
        self.thresholds_ = thresholds
        self.column_names_ = column_names
    
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", category=FutureWarning)
            X = self._validate_data(X, reset=False)

        n_samples, n_features = X.shape
        sum_of_bin_features = sum(self.n_thresholds_)
        bin_X = np.zeros((n_samples, sum_of_bin_features), dtype=np.intc)
        col_ix = 0
        for jj in range(n_features):
            if self.n_thresholds_[jj] == 0: continue
            column = X[:, jj]
            for t_ix in range(self.n_thresholds_[jj]):
                bin_X[:, col_ix] = column <= self.thresholds_[jj][t_ix]
                col_ix += 1
        column_names = itertools.chain.from_iterable(self.column_names_)
        return pd.DataFrame(bin_X, columns=column_names)
    
def _column_threshold(t):
    if int(t) == t:
        return str(t)
    if t % 1 == 0:
        return str(t)
    if math.log10(abs(t)) >= 6 or math.log10(abs(t)) <= -4:
        return f"{t:.2e}"
    if math.log10(abs(t)) >= 2:
        return f"{t:.2f}".rstrip('0').rstrip('.')
    return f"{t:f}".rstrip('0').rstrip('.')