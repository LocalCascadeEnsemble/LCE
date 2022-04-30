import math
import numbers
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ._lcetree import LCETreeClassifier, LCETreeRegressor


class LCEClassifier(ClassifierMixin, BaseEstimator):
    """
    A **Local Cascade Ensemble (LCE) classifier**. LCEClassifier is **compatible with scikit-learn**; 
    it passes the `check_estimator <https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator>`_.
    Therefore, it can interact with scikit-learn pipelines and model selection tools.
    

    Parameters
    ----------
    n_estimators : int, default=10
        The number of trees in the ensemble.
        
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        
    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported strategies
        are "best" to choose the best split and "random" to choose the best random 
        split.
    
    max_depth : int, default=2
        The maximum depth of a tree. 
        
    max_features : int, float or {"auto", "sqrt", "log"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator 
        (with replacement by default, see ``bootstrap`` for more details).
        
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus, `max_samples` should be in the interval `(0.0, 1.0]`.
        
    min_samples_leaf : int or float, default=5
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.    

    n_iter: int, default=10
        Number of iterations to set the hyperparameters of the base classifier (XGBoost)
        in Hyperopt. 
        
    metric: string, default="accuracy"
        The score of the base classifier (XGBoost) optimized by Hyperopt. Supported metrics 
        are the ones from `scikit-learn <https://scikit-learn.org/stable/modules/model_evaluation.html>`_.
        
    xgb_max_n_estimators : int, default=100
        The maximum number of XGBoost estimators. The number of estimators of 
        XGBoost corresponds to the number of boosting rounds.
        
    xgb_n_estimators_step : int, default=10
        Spacing between XGBoost n_estimators. The range of XGBoost n_estimators 
        for hyperparameter optimization (Hyperopt) is: 
        `range(1, xgb_max_n_estimators + xgb_n_estimators_step, xgb_n_estimators_step)`.

    xgb_max_depth : int, default= 10
        Maximum tree depth for XGBoost base learners. The range of XGBoost max_depth 
        for hyperparameter optimization (Hyperopt) is: `range(1, xgb_max_depth + 1)`.
        
    xgb_min_learning_rate : float, default=0.05
        Minimum learning rate of XGBoost. The learning rate corresponds to the 
        step size shrinkage used in update to prevent overfitting. After each 
        boosting step, the learning rate shrinks the feature weights to make the boosting 
        process more conservative. 
        
    xgb_max_learning_rate : float, default=0.5
        Maximum learning rate of XGBoost.
    
    xgb_learning_rate_step : float, default=0.05
        Spacing between XGBoost learning_rate. The range of XGBoost learning_rate 
        for hyperparameter optimization (Hyperopt) is: 
        `np.arange(xgb_min_learning_rate, xgb_max_learning_rate + xgb_learning_rate_step, xgb_learning_rate_step)`.
    
    xgb_booster : {"dart", "gblinear", "gbtree"}, default="gbtree"
        The type of booster to use. "gbtree" and "dart" use tree based models 
        while "gblinear" uses linear functions.
        
    xgb_min_gamma : float, default=0.05
        Minimum gamma of XGBoost. Gamma corresponds to the minimum loss reduction 
        required to make a further partition on a leaf node of the tree. 
        The larger gamma is, the more conservative XGBoost algorithm will be.
    
    xgb_max_gamma : float, default=0.5 
        Maximum gamma of XGBoost.
    
    xgb_gamma_step : float, default=0.05,
        Spacing between XGBoost gamma. The range of XGBoost gamma for hyperparameter 
        optimization (Hyperopt) is: 
        `np.arange(xgb_min_gamma, xgb_max_gamma + xgb_gamma_step, xgb_gamma_step)`.
    
    xgb_min_min_child_weight : int, default=3 
        Minimum min_child_weight of XGBoost. min_child_weight defines the
        minimum sum of instance weight (hessian) needed in a child. If the tree 
        partition step results in a leaf node with the sum of instance weight 
        less than min_child_weight, then the building process will give up further 
        partitioning. The larger min_child_weight is, the more conservative XGBoost 
        algorithm will be.
    
    xgb_max_min_child_weight : int, default=10
        Minimum min_child_weight of XGBoost.
    
    xgb_min_child_weight_step : int, default=1,
        Spacing between XGBoost min_child_weight. The range of XGBoost min_child_weight
        for hyperparameter optimization (Hyperopt) is: 
        `range(xgb_min_min_child_weight, xgb_max_min_child_weight + xgb_min_child_weight_step, xgb_min_child_weight_step)`.
        
    xgb_subsample : float, default=0.8 
        XGBoost subsample ratio of the training instances. Setting it to 0.5 means 
        that XGBoost would randomly sample half of the training data prior to 
        growing trees, and this will prevent overfitting. Subsampling will occur 
        once in every boosting iteration.
    
    xgb_colsample_bytree : float, default=0.8
        XGBoost subsample ratio of columns when constructing each tree. 
        Subsampling occurs once for every tree constructed.
    
    xgb_colsample_bylevel : float, default=1.0
        XGBoost subsample ratio of columns for each level. Subsampling occurs 
        once for every new depth level reached in a tree. Columns are subsampled 
        from the set of columns chosen for the current tree.
    
    xgb_colsample_bynode : float, default=1.0
        XGBoost subsample ratio of columns for each node (split). Subsampling 
        occurs once every time a new split is evaluated. Columns are subsampled 
        from the set of columns chosen for the current level.
        
    xgb_min_reg_alpha : float, default=0.01
        Minimum reg_alpha of XGBoost. reg_alpha corresponds to the L1 regularization 
        term on the weights. Increasing this value will make XGBoost model more 
        conservative.
    
    xgb_max_reg_alpha : float, default=0.1
        Maximum reg_alpha of XGBoost.
    
    xgb_reg_alpha_step : float, default=0.05
        Spacing between XGBoost reg_alpha. The range of XGBoost reg_alpha for 
        hyperparameter optimization (Hyperopt) is: 
        `np.arange(xgb_min_reg_alpha, xgb_max_reg_alpha + xgb_reg_alpha_step, xgb_reg_alpha_step)`.
                 
    xgb_min_reg_lambda : float, default=0.01
        Minimum reg_lambda of XGBoost. reg_lambda corresponds to the L2 regularization 
        term on the weights. Increasing this value will make XGBoost model more 
        conservative.
    
    xgb_max_reg_lambda : float, default=0.1
        Maximum reg_lambda of XGBoost.
    
    xgb_reg_lambda_step : float, default=0.05
        Spacing between XGBoost reg_lambda. The range of XGBoost reg_lambda for 
        hyperparameter optimization (Hyperopt) is: 
        `np.arange(xgb_min_reg_lambda, xgb_max_reg_lambda + xgb_reg_lambda_step, xgb_reg_lambda_step)`.
    
    n_jobs : int, default=None
        The number of jobs to run in parallel. 
        ``None`` means 1. ``-1`` means using all processors. 

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``), the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``), the base classifier (XGBoost) and
        the Hyperopt algorithm.
 
    verbose : int, default=0
        Controls the verbosity when fitting.
        
    Attributes
    ----------
    base_estimator_ : LCETreeClassifier
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of LCETreeClassifier
        The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        The classes labels.
        
    n_classes_ : int
        The number of classes.

    n_features_in_ : int
        The number of features when ``fit`` is performed.

    encoder_ : LabelEncoder
        The encoder to have target labels with value between 0 and n_classes-1.
    
    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    References
    ----------
    .. [1] Fauvel, K., E. Fromont, V. Masson, P. Faverdin and A. Termier. "XEM: An Explainable-by-Design Ensemble Method for Multivariate Time Series Classification", Data Mining and Knowledge Discovery, 2022. https://hal.inria.fr/hal-03599214/document

    Examples
    --------
    >>> from lce import LCEClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> iris = load_iris()
    >>> clf = LCEClassifier(n_jobs=-1, random_state=123)
    >>> cross_val_score(clf, iris.data, iris.target, cv=3)
    array([0.98, 0.94, 0.96])
    """

    def __init__(self, n_estimators=10, bootstrap=True, criterion='gini', splitter='best', 
                 max_depth=2, max_features=None, max_samples=1.0, min_samples_leaf=5, 
                 n_iter=10, metric='accuracy', xgb_max_n_estimators=100, xgb_n_estimators_step=10, 
                 xgb_max_depth=10, xgb_min_learning_rate=0.05, xgb_max_learning_rate=0.5, 
                 xgb_learning_rate_step=0.05, xgb_booster='gbtree', xgb_min_gamma=0.05, 
                 xgb_max_gamma=0.5, xgb_gamma_step=0.05, xgb_min_min_child_weight=3, 
                 xgb_max_min_child_weight=10, xgb_min_child_weight_step=1, xgb_subsample=0.8, 
                 xgb_colsample_bytree=0.8, xgb_colsample_bylevel=1.0, xgb_colsample_bynode=1.0,
                 xgb_min_reg_alpha=0.01, xgb_max_reg_alpha=0.1, xgb_reg_alpha_step=0.05, 
                 xgb_min_reg_lambda=0.01, xgb_max_reg_lambda=0.1, xgb_reg_lambda_step=0.05, 
                 n_jobs=None, random_state=None, verbose=0):
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_samples = max_samples
        self.min_samples_leaf = min_samples_leaf
        self.n_iter = n_iter
        self.metric = metric
        self.xgb_max_n_estimators = xgb_max_n_estimators
        self.xgb_n_estimators_step = xgb_n_estimators_step
        self.xgb_max_depth = xgb_max_depth
        self.xgb_min_learning_rate = xgb_min_learning_rate
        self.xgb_max_learning_rate = xgb_max_learning_rate
        self.xgb_learning_rate_step = xgb_learning_rate_step
        self.xgb_booster = xgb_booster
        self.xgb_min_gamma = xgb_min_gamma
        self.xgb_max_gamma = xgb_max_gamma
        self.xgb_gamma_step = xgb_gamma_step
        self.xgb_min_min_child_weight = xgb_min_min_child_weight
        self.xgb_max_min_child_weight = xgb_max_min_child_weight
        self.xgb_min_child_weight_step = xgb_min_child_weight_step        
        self.xgb_subsample = xgb_subsample
        self.xgb_colsample_bytree = xgb_colsample_bytree
        self.xgb_colsample_bylevel = xgb_colsample_bylevel
        self.xgb_colsample_bynode = xgb_colsample_bynode
        self.xgb_min_reg_alpha = xgb_min_reg_alpha
        self.xgb_max_reg_alpha = xgb_max_reg_alpha
        self.xgb_reg_alpha_step = xgb_reg_alpha_step
        self.xgb_min_reg_lambda = xgb_min_reg_lambda
        self.xgb_max_reg_lambda = xgb_max_reg_lambda
        self.xgb_reg_lambda_step = xgb_reg_lambda_step
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        
    def _generate_estimator(self):
        """Generate an estimator."""
        est = LCETreeClassifier()
        est.n_classes_in = self.n_classes_
        est.criterion = self.criterion
        est.splitter = self.splitter
        est.max_depth = self.max_depth
        est.max_features = self.max_features
        est.min_samples_leaf = self.min_samples_leaf
        est.n_iter = self.n_iter
        est.metric = self.metric
        est.xgb_max_n_estimators = self.xgb_max_n_estimators
        est.xgb_n_estimators_step = self.xgb_n_estimators_step
        est.xgb_max_depth = self.xgb_max_depth
        est.xgb_min_learning_rate = self.xgb_min_learning_rate
        est.xgb_max_learning_rate = self.xgb_max_learning_rate
        est.xgb_learning_rate_step = self.xgb_learning_rate_step
        est.xgb_booster = self.xgb_booster
        est.xgb_min_gamma = self.xgb_min_gamma
        est.xgb_max_gamma = self.xgb_max_gamma
        est.xgb_gamma_step = self.xgb_gamma_step
        est.xgb_min_min_child_weight = self.xgb_min_min_child_weight
        est.xgb_max_min_child_weight = self.xgb_max_min_child_weight
        est.xgb_min_child_weight_step = self.xgb_min_child_weight_step
        est.xgb_subsample = self.xgb_subsample
        est.xgb_colsample_bytree = self.xgb_colsample_bytree
        est.xgb_colsample_bylevel = self.xgb_colsample_bylevel
        est.xgb_colsample_bynode = self.xgb_colsample_bynode
        est.xgb_min_reg_alpha = self.xgb_min_reg_alpha
        est.xgb_max_reg_alpha = self.xgb_max_reg_alpha
        est.xgb_reg_alpha_step = self.xgb_reg_alpha_step
        est.xgb_min_reg_lambda = self.xgb_min_reg_lambda
        est.xgb_max_reg_lambda = self.xgb_max_reg_lambda
        est.xgb_reg_lambda_step = self.xgb_reg_lambda_step
        est.n_jobs = self.n_jobs
        est.random_state = self.random_state
        est.verbose = self.verbose
        return est
    
    
    def _more_tags(self):
        """Update scikit-learn estimator tags."""
        return {'allow_nan': True,
                'requires_y': True}
    
    
    def _validate_extra_parameters(self, X):  
        """Validate parameters not already validated by methods employed."""
        # Validate max_depth
        if isinstance(self.max_depth, numbers.Integral):
            if not (0 <= self.max_depth):
                raise ValueError("max_depth must be greater than or equal to 0, "
                                 "got {0}.".format(self.max_depth))
        else:
            raise ValueError("max_depth must be int")
            
        # Validate min_samples_leaf
        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
        elif isinstance(self.min_samples_leaf, float):
            if not 0. < self.min_samples_leaf <= 0.5:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            self.min_samples_leaf = int(math.ceil(self.min_samples_leaf * X.shape[0]))
        else:
             raise ValueError("min_samples_leaf must be int or float")
        
        # Validate n_iter
        if isinstance(self.n_iter, numbers.Integral):
            if self.n_iter <= 0:
                raise ValueError("n_iter must be greater than 0, "
                                 "got {0}.".format(self.n_iter))
        else:
            raise ValueError("n_iter must be int")
            
        # Validate verbose
        if isinstance(self.verbose, numbers.Integral):
            if self.verbose < 0:
                raise ValueError("verbose must be greater than or equal to 0, "
                                 "got {0}.".format(self.verbose))
        else:
            raise ValueError("verbose must be int")
    

    def fit(self, X, y):
        """
        Build a forest of LCE trees from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, force_all_finite='allow-nan')
        check_classification_targets(y)
        self._validate_extra_parameters(X)
        self.n_features_in_ = X.shape[1]
        self.X_ = True
        self.y_ = True
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = np.unique(y).size
        self.encoder_ = LabelEncoder()
        self.encoder_.fit(self.classes_)
        self.base_estimator_ = self._generate_estimator()
        self.estimators_ = BaggingClassifier(base_estimator=self.base_estimator_, 
                                             n_estimators=self.n_estimators,
                                             bootstrap=self.bootstrap, 
                                             max_samples=self.max_samples, 
                                             n_jobs=self.n_jobs, 
                                             random_state = self.random_state)
        self.estimators_.fit(X, y)
        return self
    

    def predict(self, X):
        """
        Predict class for X.
        The predicted class of an input sample is computed as the class with 
        the highest mean predicted probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X, force_all_finite='allow-nan')
        predictions = self.estimators_.predict(X)
        return self.encoder_.inverse_transform(predictions)
    
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        The predicted class probabilities of an input sample is computed as 
        the mean predicted class probabilities of the base estimators in the 
        ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The class probabilities of the input samples. The order of the 
            classes corresponds to that in the attribute ``classes_``.
        """
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X, force_all_finite='allow-nan')
        return self.estimators_.predict_proba(X)
        
    
    def set_params(self, **params):
        """
        Set the parameters of the estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
        """
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value            
        
        return self
    
    
class LCERegressor(RegressorMixin, BaseEstimator):
    """
    A **Local Cascade Ensemble (LCE) regressor**. LCERegressor is **compatible with scikit-learn**; 
    it passes the `check_estimator <https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html#sklearn.utils.estimator_checks.check_estimator>`_.
    Therefore, it can interact with scikit-learn pipelines and model selection tools.
    

    Parameters
    ----------
    n_estimators : int, default=10
        The number of trees in the ensemble.
        
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.

    criterion : {"squared_error", "friedman_mse", "absolute_error", "poisson"}, default="squared_error"
        The function to measure the quality of a split. Supported criteria are "squared_error" for 
        the mean squared error, which is equal to variance reduction as feature selection 
        criterion and minimizes the L2 loss using the mean of each terminal node, 
        "friedman_mse", which uses mean squared error with Friedman's improvement score 
        for potential splits, "absolute_error" for the mean absolute error, which 
        minimizes the L1 loss using the median of each terminal node, and "poisson" 
        which uses reduction in Poisson deviance to find splits.
        
    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported strategies
        are "best" to choose the best split and "random" to choose the best random 
        split.
    
    max_depth : int, default=2
        The maximum depth of a tree. 
        
    max_features : int, float or {"auto", "sqrt", "log"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=sqrt(n_features)`.
        - If "sqrt", then `max_features=sqrt(n_features)` (same as "auto").
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator 
        (with replacement by default, see ``bootstrap`` for more details).
        
        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples. Thus, `max_samples` should be in the interval `(0.0, 1.0]`.
        
    min_samples_leaf : int or float, default=5
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.    

    n_iter: int, default=10
        Number of iterations to set the hyperparameters of the base regressor (XGBoost) 
        in Hyperopt.
        
    metric: string, default="neg_mean_squared_error"
        The score of the base regressor (XGBoost) optimized by Hyperopt. Supported metrics 
        are the ones from `scikit-learn <https://scikit-learn.org/stable/modules/model_evaluation.html>`_.

    xgb_max_n_estimators : int, default=100
        The maximum number of XGBoost estimators. The number of estimators of 
        XGBoost corresponds to the number of boosting rounds.
        
    xgb_n_estimators_step : int, default=10
        Spacing between XGBoost n_estimators. The range of XGBoost n_estimators 
        for hyperparameter optimization (Hyperopt) is: 
        `range(1, xgb_max_n_estimators + xgb_n_estimators_step, xgb_n_estimators_step)`.

    xgb_max_depth : int, default= 10
        Maximum tree depth for XGBoost base learners. The range of XGBoost max_depth 
        for hyperparameter optimization (Hyperopt) is: `range(1, xgb_max_depth + 1)`.
        
    xgb_min_learning_rate : float, default=0.05
        Minimum learning rate of XGBoost. The learning rate corresponds to the 
        step size shrinkage used in update to prevent overfitting. After each 
        boosting step, the learning rate shrinks the feature weights to make the boosting 
        process more conservative. 
        
    xgb_max_learning_rate : float, default=0.5
        Maximum learning rate of XGBoost.
    
    xgb_learning_rate_step : float, default=0.05
        Spacing between XGBoost learning_rate. The range of XGBoost learning_rate 
        for hyperparameter optimization (Hyperopt) is: 
        `np.arange(xgb_min_learning_rate, xgb_max_learning_rate + xgb_learning_rate_step, xgb_learning_rate_step)`.
    
    xgb_booster : {"dart", "gblinear", "gbtree"}, default="gbtree"
        The type of booster to use. "gbtree" and "dart" use tree based models 
        while "gblinear" uses linear functions.
        
    xgb_min_gamma : float, default=0.05
        Minimum gamma of XGBoost. Gamma corresponds to the minimum loss reduction 
        required to make a further partition on a leaf node of the tree. 
        The larger gamma is, the more conservative XGBoost algorithm will be.
    
    xgb_max_gamma : float, default=0.5 
        Maximum gamma of XGBoost.
    
    xgb_gamma_step : float, default=0.05,
        Spacing between XGBoost gamma. The range of XGBoost gamma for hyperparameter 
        optimization (Hyperopt) is: 
        `np.arange(xgb_min_gamma, xgb_max_gamma + xgb_gamma_step, xgb_gamma_step)`.
    
    xgb_min_min_child_weight : int, default=3 
        Minimum min_child_weight of XGBoost. min_child_weight defines the
        minimum sum of instance weight (hessian) needed in a child. If the tree 
        partition step results in a leaf node with the sum of instance weight 
        less than min_child_weight, then the building process will give up further 
        partitioning. The larger min_child_weight is, the more conservative XGBoost 
        algorithm will be.
    
    xgb_max_min_child_weight : int, default=10
        Minimum min_child_weight of XGBoost.
    
    xgb_min_child_weight_step : int, default=1,
        Spacing between XGBoost min_child_weight. The range of XGBoost min_child_weight
        for hyperparameter optimization (Hyperopt) is: 
        `range(xgb_min_min_child_weight, xgb_max_min_child_weight + xgb_min_child_weight_step, xgb_min_child_weight_step)`.
        
    xgb_subsample : float, default=0.8 
        XGBoost subsample ratio of the training instances. Setting it to 0.5 means 
        that XGBoost would randomly sample half of the training data prior to 
        growing trees, and this will prevent overfitting. Subsampling will occur 
        once in every boosting iteration.
    
    xgb_colsample_bytree : float, default=0.8
        XGBoost subsample ratio of columns when constructing each tree. 
        Subsampling occurs once for every tree constructed.
    
    xgb_colsample_bylevel : float, default=1.0
        XGBoost subsample ratio of columns for each level. Subsampling occurs 
        once for every new depth level reached in a tree. Columns are subsampled 
        from the set of columns chosen for the current tree.
    
    xgb_colsample_bynode : float, default=1.0
        XGBoost subsample ratio of columns for each node (split). Subsampling 
        occurs once every time a new split is evaluated. Columns are subsampled 
        from the set of columns chosen for the current level.
        
    xgb_min_reg_alpha : float, default=0.01
        Minimum reg_alpha of XGBoost. reg_alpha corresponds to the L1 regularization 
        term on the weights. Increasing this value will make XGBoost model more 
        conservative.
    
    xgb_max_reg_alpha : float, default=0.1
        Maximum reg_alpha of XGBoost.
    
    xgb_reg_alpha_step : float, default=0.05
        Spacing between XGBoost reg_alpha. The range of XGBoost reg_alpha for 
        hyperparameter optimization (Hyperopt) is: 
        `np.arange(xgb_min_reg_alpha, xgb_max_reg_alpha + xgb_reg_alpha_step, xgb_reg_alpha_step)`.
                 
    xgb_min_reg_lambda : float, default=0.01
        Minimum reg_lambda of XGBoost. reg_lambda corresponds to the L2 regularization 
        term on the weights. Increasing this value will make XGBoost model more 
        conservative.
    
    xgb_max_reg_lambda : float, default=0.1
        Maximum reg_lambda of XGBoost.
    
    xgb_reg_lambda_step : float, default=0.05
        Spacing between XGBoost reg_lambda. The range of XGBoost reg_lambda for 
        hyperparameter optimization (Hyperopt) is: 
        `np.arange(xgb_min_reg_lambda, xgb_max_reg_lambda + xgb_reg_lambda_step, xgb_reg_lambda_step)`.
    
    n_jobs : int, default=None
        The number of jobs to run in parallel. 
        ``None`` means 1. ``-1`` means using all processors. 

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``), the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``), the base classifier (XGBoost) and
        the Hyperopt algorithm.
 
    verbose : int, default=0
        Controls the verbosity when fitting.
        
    Attributes
    ----------
    base_estimator_ : LCETreeRegressor
        The child estimator template used to create the collection of fitted
        sub-estimators.

    estimators_ : list of LCETreeRegressor
        The collection of fitted sub-estimators.

    n_features_in_ : int
        The number of features when ``fit`` is performed.
    
    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    Examples
    --------
    >>> from lce import LCERegressor
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import cross_val_score
    >>> diabetes = load_diabetes()
    >>> reg = LCERegressor(n_jobs=-1, random_state=0)
    >>> cross_val_score(reg, diabetes.data, diabetes.target, cv=3)
    array([0.35317032, 0.39600438, 0.33379507])
    """

    def __init__(self, n_estimators=10, bootstrap=True, criterion='squared_error', splitter='best', 
                 max_depth=2, max_features=None, max_samples=1.0, min_samples_leaf=5, 
                 metric='neg_mean_squared_error', n_iter=10, xgb_max_n_estimators=100, 
                 xgb_n_estimators_step=10, xgb_max_depth=10, xgb_min_learning_rate=0.05, 
                 xgb_max_learning_rate=0.5, xgb_learning_rate_step=0.05, xgb_booster='gbtree', 
                 xgb_min_gamma=0.05, xgb_max_gamma=0.5, xgb_gamma_step=0.05, xgb_min_min_child_weight=3, 
                 xgb_max_min_child_weight=10, xgb_min_child_weight_step=1, xgb_subsample=0.8, 
                 xgb_colsample_bytree=0.8, xgb_colsample_bylevel=1.0, xgb_colsample_bynode=1.0,
                 xgb_min_reg_alpha=0.01, xgb_max_reg_alpha=0.1, xgb_reg_alpha_step=0.05, 
                 xgb_min_reg_lambda=0.01, xgb_max_reg_lambda=0.1, xgb_reg_lambda_step=0.05, 
                 n_jobs=None, random_state=None, verbose=0):
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_samples = max_samples
        self.min_samples_leaf = min_samples_leaf
        self.n_iter = n_iter
        self.metric = metric
        self.xgb_max_n_estimators = xgb_max_n_estimators
        self.xgb_n_estimators_step = xgb_n_estimators_step
        self.xgb_max_depth = xgb_max_depth
        self.xgb_min_learning_rate = xgb_min_learning_rate
        self.xgb_max_learning_rate = xgb_max_learning_rate
        self.xgb_learning_rate_step = xgb_learning_rate_step
        self.xgb_booster = xgb_booster
        self.xgb_min_gamma = xgb_min_gamma
        self.xgb_max_gamma = xgb_max_gamma
        self.xgb_gamma_step = xgb_gamma_step
        self.xgb_min_min_child_weight = xgb_min_min_child_weight
        self.xgb_max_min_child_weight = xgb_max_min_child_weight
        self.xgb_min_child_weight_step = xgb_min_child_weight_step        
        self.xgb_subsample = xgb_subsample
        self.xgb_colsample_bytree = xgb_colsample_bytree
        self.xgb_colsample_bylevel = xgb_colsample_bylevel
        self.xgb_colsample_bynode = xgb_colsample_bynode
        self.xgb_min_reg_alpha = xgb_min_reg_alpha
        self.xgb_max_reg_alpha = xgb_max_reg_alpha
        self.xgb_reg_alpha_step = xgb_reg_alpha_step
        self.xgb_min_reg_lambda = xgb_min_reg_lambda
        self.xgb_max_reg_lambda = xgb_max_reg_lambda
        self.xgb_reg_lambda_step = xgb_reg_lambda_step
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        
    def _generate_estimator(self):
        """Generate an estimator."""
        est = LCETreeRegressor()
        est.criterion = self.criterion
        est.splitter = self.splitter
        est.max_depth = self.max_depth
        est.max_features = self.max_features
        est.min_samples_leaf = self.min_samples_leaf
        est.n_iter = self.n_iter
        est.metric = self.metric
        est.xgb_max_n_estimators = self.xgb_max_n_estimators
        est.xgb_n_estimators_step = self.xgb_n_estimators_step
        est.xgb_max_depth = self.xgb_max_depth
        est.xgb_min_learning_rate = self.xgb_min_learning_rate
        est.xgb_max_learning_rate = self.xgb_max_learning_rate
        est.xgb_learning_rate_step = self.xgb_learning_rate_step
        est.xgb_booster = self.xgb_booster
        est.xgb_min_gamma = self.xgb_min_gamma
        est.xgb_max_gamma = self.xgb_max_gamma
        est.xgb_gamma_step = self.xgb_gamma_step
        est.xgb_min_min_child_weight = self.xgb_min_min_child_weight
        est.xgb_max_min_child_weight = self.xgb_max_min_child_weight
        est.xgb_min_child_weight_step = self.xgb_min_child_weight_step
        est.xgb_subsample = self.xgb_subsample
        est.xgb_colsample_bytree = self.xgb_colsample_bytree
        est.xgb_colsample_bylevel = self.xgb_colsample_bylevel
        est.xgb_colsample_bynode = self.xgb_colsample_bynode
        est.xgb_min_reg_alpha = self.xgb_min_reg_alpha
        est.xgb_max_reg_alpha = self.xgb_max_reg_alpha
        est.xgb_reg_alpha_step = self.xgb_reg_alpha_step
        est.xgb_min_reg_lambda = self.xgb_min_reg_lambda
        est.xgb_max_reg_lambda = self.xgb_max_reg_lambda
        est.xgb_reg_lambda_step = self.xgb_reg_lambda_step
        est.n_jobs = self.n_jobs
        est.random_state = self.random_state
        est.verbose = self.verbose
        return est
    
    
    def _more_tags(self):
        """Update scikit-learn estimator tags."""
        return {'allow_nan': True,
                'requires_y': True}
    
    
    def _validate_extra_parameters(self, X):  
        """Validate parameters not already validated by methods employed."""
        # Validate max_depth
        if isinstance(self.max_depth, numbers.Integral):
            if not (0 <= self.max_depth):
                raise ValueError("max_depth must be greater than or equal to 0, "
                                 "got {0}.".format(self.max_depth))
        else:
            raise ValueError("max_depth must be int")
            
        # Validate min_samples_leaf
        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
        elif isinstance(self.min_samples_leaf, float):
            if not 0. < self.min_samples_leaf <= 0.5:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            self.min_samples_leaf = int(math.ceil(self.min_samples_leaf * X.shape[0]))
        else:
             raise ValueError("min_samples_leaf must be int or float")
        
        # Validate n_iter
        if isinstance(self.n_iter, numbers.Integral):
            if self.n_iter <= 0:
                raise ValueError("n_iter must be greater than 0, "
                                 "got {0}.".format(self.n_iter))
        else:
            raise ValueError("n_iter must be int")
            
        # Validate verbose
        if isinstance(self.verbose, numbers.Integral):
            if self.verbose < 0:
                raise ValueError("verbose must be greater than or equal to 0, "
                                 "got {0}.".format(self.verbose))
        else:
            raise ValueError("verbose must be int")
            

    def fit(self, X, y):
        """
        Build a forest of LCE trees from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (real numbers).

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, y_numeric=True, force_all_finite='allow-nan')
        self._validate_extra_parameters(X)
        self.n_features_in_ = X.shape[1]
        self.X_ = True
        self.y_ = True
        self.base_estimator_ = self._generate_estimator()
        self.estimators_ = BaggingRegressor(base_estimator=self.base_estimator_, 
                                             n_estimators=self.n_estimators,
                                             bootstrap=self.bootstrap, 
                                             max_samples=self.max_samples, 
                                             n_jobs=self.n_jobs, 
                                             random_state = self.random_state)
        self.estimators_.fit(X, y)
        return self
    

    def predict(self, X):
        """
        Predict regression target for X.
        The predicted regression target of an input sample is computed as the 
        mean predicted regression targets of the trees in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X, force_all_finite='allow-nan')
        return self.estimators_.predict(X)
        
    
    def set_params(self, **params):
        """
        Set the parameters of the estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
        """
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value            
        
        return self