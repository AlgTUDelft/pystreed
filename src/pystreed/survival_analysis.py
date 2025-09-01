from pystreed.base import BaseSTreeDSolver
from pystreed.data import SAData
from sklearn.utils.validation import check_is_fitted
from sksurv.util import check_array_survival
from sksurv.tree._criterion import get_unique_times
from sksurv.tree.tree import _array_to_step_function
from typing import Optional
import numpy as np
import math
from pathlib import Path
import os


class STreeDSurvivalAnalysis(BaseSTreeDSolver):
    """
    An optimal survival tree learner using dynamic programming

    This class follows the design of the scikit-survival package.
    See https://scikit-survival.readthedocs.io/en/stable/user_guide/00-introduction.html for an introduction
    Comments and method definitions taken from https://github.com/sebp/scikit-survival/blob/v0.21.0/sksurv/tree/tree.py
    """

    def __init__(self,
                 max_depth : int = 3,
                 max_num_nodes : Optional[int] = None,
                 min_leaf_node_size : int = 1,
                 time_limit : float = 600,
                 hyper_tune : bool = False,
                 use_branch_caching: bool = False,
                 use_dataset_caching: bool = True,
                 use_terminal_solver: bool = True,
                 use_upper_bound: bool = True,
                 use_lower_bound: bool = True,
                 upper_bound: float = 2**31 - 1,
                 verbose : bool = False,
                 random_seed: int = 27, 
                 continuous_binarize_strategy: str = 'quantile',
                 n_thresholds: int = 5,
                 n_categories: int = 5):
        BaseSTreeDSolver.__init__(self, "survival-analysis",
            max_depth=max_depth,
            max_num_nodes=max_num_nodes,
            min_leaf_node_size=min_leaf_node_size,
            time_limit=time_limit,
            feature_ordering="in-order",
            hyper_tune = hyper_tune,
            use_branch_caching=use_branch_caching,
            use_dataset_caching=use_dataset_caching,
            use_terminal_solver=use_terminal_solver,
            use_similarity_lower_bound=False,
            use_upper_bound=use_upper_bound,
            use_lower_bound=use_lower_bound,
            upper_bound=upper_bound,
            verbose=verbose,
            random_seed=random_seed,
            continuous_binarize_strategy=continuous_binarize_strategy,
            n_thresholds=n_thresholds,
            n_categories=n_categories)
        self._label_type = np.double
    """
        Construct a STreeDSurvivalAnalysis solver

        Parameters:
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
            use_upper_bound: Enable/Disable the use of upper bounds (Enabled is typically faster)
            use_lower_bound: Enable/Disable the use of lower bounds (Enabled is typically faster)
            upper_bound: Search for a tree better than the provided upper bound
            verbose: Enable/Disable verbose output
            random_seed: the random seed used by the solver (for example when creating folds)
            continuous_binarization_strategy: the strategy used for binarizing continuous features
            n_thresholds: the number of thresholds to use per continuous feature
            n_categories: the number of categories to use per categorical feature
        """

    def _process_extra_data(self, X, events):
        """
        Preprocess the events into a list of SAData objects. Initialize the hazard value with -1
        """
        if events is None:
            return [SAData(1, -1.0) for x in X]    
        return [SAData(e, -1.0) for e in events]

    def fit(self, X, y, categorical=None):
        """
        Fits a STreeD survival tree model to the given training data.

         Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data matrix

        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        categorical : array-like, 
            List of column names that are categorical

        Returns
        -------
        self
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError('x and y have different number of rows')
        events, times = check_array_survival(X, y)
        self._baseline_hazard_function = self._compute_baseline_cumulative_hazard_function(events, times)
        self.unique_times_, self.is_event_time_ = get_unique_times(times, events)
        self.n_outputs_ = self.unique_times_.shape[0]
        self.n_classes_ = np.ones(self.n_outputs_, dtype=np.intp) * 2

        return super().fit(X, times, events, categorical)
    
    def score(self, X, y_true):
        """
        Computes the score for the given input feature data and the true

        Args:
            x : array-like, shape = (n_samples, n_features)
            Data matrix
            
            y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.

        Returns:
            The score
        """
        check_is_fitted(self, "fit_result")
        if X.shape[0] != y_true.shape[0]:
            raise ValueError('x and y have different number of rows')
        events, times = check_array_survival(X, y_true)
        return super().score(X, times, events)

    def predict(self, X, extra_data=None):
        """Predict risk score.

        The risk score is the total number of events, which can
        be estimated by the sum of the estimated cumulative
        hazard function :math:`\\hat{H}_h` in terminal node :math:`h`.

        .. math::

            \\sum_{j=1}^{n(h)} \\hat{H}_h(T_{j} \\mid x) ,

        where :math:`n(h)` denotes the number of distinct event times
        of samples belonging to the same terminal node as :math:`x`.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = (n_samples, n_features)
            Data matrix.

        check_input : boolean, default: True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        risk_scores : ndarray, shape = (n_samples,)
            Predicted risk scores.
        """
        chf = self.predict_cumulative_hazard_function(X, return_array=True)
        return chf[:, self.is_event_time_].sum(1)

    def _compute_baseline_cumulative_hazard_function(self, events, times):
        """
        Compute an estimate for the baseline cumulative hazard function
        using the Nelson-Aalen estimator
        """
        ts = {}
        n = len(events)
        for d, t in zip(events, times):
            if t not in ts:
                ts[t] = [0, 0]
            ts[t][d] += 1

        d = {0: 1 / (n + 1)}
        at_risk = n
        sum = 0
        for t in sorted(ts.keys()):
            left, died = ts[t]
            sum += died / at_risk
            at_risk -= died + left

            if died > 0:
                d[t] = sum

        keys = [*d.keys()]

        def f(x):
            try:
                return d[x]
            except:
                pass

            a = 0
            b = len(keys) - 1
            while a != b:
                mid = (a + b + 1) // 2
                if keys[mid] > x:
                    b = mid - 1
                else:
                    a = mid
            return d[keys[a]]

        return f

    def predict_cumulative_hazard_function(self, X, extra_data=None, return_array=False):
        """Predict cumulative hazard function.

        The cumulative hazard function (CHF) for an individual
        with feature vector :math:`x` is computed from
        all samples of the training data that are in the
        same terminal node as :math:`x`.
        It is estimated by the Nelson–Aalen estimator.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = (n_samples, n_features)
            Data matrix.

        check_input : boolean, default: True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        return_array : boolean, default: False
            If set, return an array with the cumulative hazard rate
            for each `self.unique_times_`, otherwise an array of
            :class:`sksurv.functions.StepFunction`.

        Returns
        -------
        cum_hazard : ndarray
            If `return_array` is set, an array with the cumulative hazard rate
            for each `self.unique_times_`, otherwise an array of length `n_samples`
            of :class:`sksurv.functions.StepFunction` instances will be returned.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sksurv.datasets import load_whas500
        >>> from sksurv.tree import SurvivalTree

        Load and prepare the data.

        >>> X, y = load_whas500()
        >>> X = X.astype(float)

        Fit the model.

        >>> estimator = SurvivalTree().fit(X, y)

        Estimate the cumulative hazard function for the first 5 samples.

        >>> chf_funcs = estimator.predict_cumulative_hazard_function(X.iloc[:5])

        Plot the estimated cumulative hazard functions.

        >>> for fn in chf_funcs:
        ...    plt.step(fn.x, fn(fn.x), where="post")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.show()
        """
        thetas = super().predict(X, extra_data)
        arr = np.array([
                [theta * self._baseline_hazard_function(t) for t in self.unique_times_]
                 for theta in thetas])
        if return_array: return arr
        return _array_to_step_function(self.unique_times_, arr)

    def predict_survival_function(self, X, extra_data=None, return_array=False):
        """Predict survival function.

        The survival function for an individual
        with feature vector :math:`x` is computed from
        all samples of the training data that are in the
        same terminal node as :math:`x`.
        It is estimated by the Kaplan-Meier estimator.

        Parameters
        ----------
        X : array-like or sparse matrix, shape = (n_samples, n_features)
            Data matrix.

        check_input : boolean, default: True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        return_array : boolean, default: False
            If set, return an array with the probability
            of survival for each `self.unique_times_`,
            otherwise an array of :class:`sksurv.functions.StepFunction`.

        Returns
        -------
        survival : ndarray
            If `return_array` is set, an array with the probability of
            survival for each `self.unique_times_`, otherwise an array of
            length `n_samples` of :class:`sksurv.functions.StepFunction`
            instances will be returned.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from sksurv.datasets import load_whas500
        >>> from sksurv.tree import SurvivalTree

        Load and prepare the data.

        >>> X, y = load_whas500()
        >>> X = X.astype(float)

        Fit the model.

        >>> estimator = SurvivalTree().fit(X, y)

        Estimate the survival function for the first 5 samples.

        >>> surv_funcs = estimator.predict_survival_function(X.iloc[:5])

        Plot the estimated survival functions.

        >>> for fn in surv_funcs:
        ...    plt.step(fn.x, fn(fn.x), where="post")
        ...
        >>> plt.ylim(0, 1)
        >>> plt.show()
        """
        thetas = super().predict(X, extra_data)
        arr = np.array([
                [math.exp(-theta * self._baseline_hazard_function(t)) for t in self.unique_times_]
                 for theta in thetas])
        if return_array: return arr
        return _array_to_step_function(self.unique_times_, arr)
    

    def _export_dot_leaf_node(self, fh, node, node_id, label_names, train_data):
        try:
            import matplotlib.pyplot as plt
            theta = node.label
            arr = np.array([[math.exp(-theta * self._baseline_hazard_function(t)) for t in self.unique_times_]])
            fn = _array_to_step_function(self.unique_times_, arr)[0]
            plt.figure(figsize=(2,2))
            plt.step(fn.x, fn(fn.x), where="post")
            plt.ylim(0, 1)
            plt.ylabel("Probability of survival $\hat{S}(t)$")
            plt.xlabel("time $t$")
            filename = self.__export_directory / f"leaf_KM_{node_id}.png"
            if not os.path.exists(self.__export_directory):
                os.makedirs(self.__export_directory)
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            fh.write(f"{node_id}  [label=\"\", image=\"{filename}\", width=2, height=2, fixedsize=true] ;\n")
        except:
            return super()._export_dot_leaf_node(fh, node, node_id, label_names, train_data)


    def export_dot(self, filename, feature_names=None, label_names=None):
        """
        Export the tree as a .dot file (for plotting)
        """
        self.__export_directory = Path(filename).parent.absolute() / "leaf-distributions"
        return super().export_dot(filename, feature_names, label_names)
        
