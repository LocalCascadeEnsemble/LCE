import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ._xgboost import xgb_opt_classifier, xgb_opt_regressor


class LCETreeClassifier(ClassifierMixin, BaseEstimator):
    """
    A LCE Tree classifier.
    

    Parameters
    ----------  
    n_classes_in : int, default=None
        The number of classes from the input data.
    
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
        Number of iterations to set the hyperparameters of the base classifier 
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
        range(1, xgb_max_n_estimators+xgb_n_estimators_step, xgb_n_estimators_step).

    xgb_max_depth : int, default= 10
        Maximum tree depth for XGBoost base learners. The range of XGBoost max_depth 
        for Hyperopt is: range(1, xgb_max_depth+1).
        
    xgb_min_learning_rate : float, default=0.05
        Minimum learning rate of XGBoost. The learning rate corresponds to the 
        step size shrinkage used in update to prevent overfitting. After each 
        boosting step, we can directly get the weights of new features, 
        and the learning rate shrinks the feature weights to make the boosting 
        process more conservative. 
        
    xgb_max_learning_rate : float, default=0.5
        Maximum learning rate of XGBoost.
    
    xgb_learning_rate_step : float, default=0.05
        Spacing between XGBoost learning_rate. The range of XGBoost learning_rate 
        for hyperparameter optimization (Hyperopt) is: 
        np.arange(xgb_min_learning_rate, xgb_max_learning_rate+xgb_learning_rate_step, xgb_learning_rate_step).
    
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
        np.arange(xgb_min_gamma, xgb_max_gamma+xgb_gamma_step, xgb_gamma_step).
    
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
        range(xgb_min_min_child_weight, xgb_max_min_child_weight+xgb_min_child_weight_step, xgb_min_child_weight_step).
        
    xgb_subsample : float, default=0.8 
        XGBoost subsample ratio of the training instances. Setting it to 0.5 means 
        that XGBoost would randomly sample half of the training data prior to 
        growing trees. and this will prevent overfitting. Subsampling will occur 
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
        np.arange(xgb_min_reg_alpha, xgb_max_reg_alpha+xgb_reg_alpha_step, xgb_reg_alpha_step).
                 
    xgb_min_reg_lambda : float, default=0.01
        Minimum reg_lambda of XGBoost. reg_lambda corresponds to the L2 regularization 
        term on the weights. Increasing this value will make XGBoost model more 
        conservative.
    
    xgb_max_reg_lambda : float, default=0.1
        Maximum reg_lambda of XGBoost.
    
    xgb_reg_lambda_step : float, default=0.05
        Spacing between XGBoost reg_lambda. The range of XGBoost reg_lambda for 
        hyperparameter optimization (Hyperopt) is: 
        np.arange(xgb_min_reg_lambda, xgb_max_reg_lambda+xgb_reg_lambda_step, xgb_reg_lambda_step).
        
    n_jobs : int, default=None
        The number of jobs to run in parallel. 
        ``None`` means 1. ``-1`` means using all processors. 

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the sampling of the features to consider when 
        looking for the best split at each node (if ``max_features < n_features``), 
        the base classifier (XGBoost) and the Hyperopt algorithm.
 
    verbose : int, default=0
        Controls the verbosity when fitting.
        
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or a list of such arrays
        The classes labels.

    n_features_in_ : int
        The number of features when ``fit`` is performed.
    """

    def __init__(self, n_classes_in=None, criterion='gini', splitter='best', max_depth=2, 
                 max_features=None, min_samples_leaf=5, n_iter=10, metric='accuracy',
                 xgb_max_n_estimators=100, xgb_n_estimators_step=10, xgb_max_depth=10,
                 xgb_min_learning_rate=0.05, xgb_max_learning_rate=0.5, xgb_learning_rate_step=0.05, 
                 xgb_booster='gbtree', xgb_min_gamma=0.05, xgb_max_gamma=0.5, xgb_gamma_step=0.05,
                 xgb_min_min_child_weight=3, xgb_max_min_child_weight=10, xgb_min_child_weight_step=1, 
                 xgb_subsample=0.8, xgb_colsample_bytree=0.8,
                 xgb_colsample_bylevel=1.0, xgb_colsample_bynode=1.0, 
                 xgb_min_reg_alpha=0.01, xgb_max_reg_alpha=0.1, xgb_reg_alpha_step=0.05, 
                 xgb_min_reg_lambda=0.01, xgb_max_reg_lambda=0.1, xgb_reg_lambda_step=0.05,
                 n_jobs=None, random_state=None, verbose=0):
        self.n_classes_in = n_classes_in
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.max_features = max_features
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
        
    
    def fit(self, X, y):
        """
        Build a LCE tree from the training set (X, y).

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
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]        
        
        def _build_tree(X, y):
            """Build a LCE tree."""
            global index_node_global

            def _create_node(X, y, depth, container):
                """Create a node in the tree."""
                # Add XGBoost predictions as features to the dataset
                model_node = xgb_opt_classifier(X, y, n_iter=self.n_iter,
                                                metric = self.metric,
                                                n_estimators=self.xgb_max_n_estimators,
                                                n_estimators_step = self.xgb_n_estimators_step,
                                                max_depth=self.xgb_max_depth,
                                                min_learning_rate = self.xgb_min_learning_rate,
                                                max_learning_rate = self.xgb_max_learning_rate,
                                                learning_rate_step = self.xgb_learning_rate_step,
                                                booster = self.xgb_booster,
                                                min_gamma = self.xgb_min_gamma,
                                                max_gamma = self.xgb_max_gamma,
                                                gamma_step = self.xgb_gamma_step,
                                                min_min_child_weight = self.xgb_min_min_child_weight,
                                                max_min_child_weight = self.xgb_max_min_child_weight,
                                                min_child_weight_step = self.xgb_min_child_weight_step,
                                                subsample=self.xgb_subsample, 
                                                colsample_bytree = self.xgb_colsample_bytree,
                                                colsample_bylevel = self.xgb_colsample_bylevel,
                                                colsample_bynode = self.xgb_colsample_bynode,
                                                min_reg_alpha = self.xgb_min_reg_alpha,
                                                max_reg_alpha = self.xgb_max_reg_alpha,
                                                reg_alpha_step = self.xgb_reg_alpha_step,
                                                min_reg_lambda  = self.xgb_min_reg_lambda,
                                                max_reg_lambda  = self.xgb_max_reg_lambda,
                                                reg_lambda_step = self.xgb_reg_lambda_step,
                                                random_state=self.random_state)
                pred_proba = np.around(model_node.predict_proba(X), 6)

                c = 0
                for i in range(0, self.n_classes_in):
                    X = np.insert(X, X.shape[1], 0, axis=1)
                    if i in y:
                        if np.unique(y).size == 1:
                            X[:,-1] = pred_proba[:,1]
                        else:
                            X[:,-1] = pred_proba[:,c]
                            c += 1
                
                # Missing data information
                num_nans = np.isnan(X).any(axis=1).sum()
                if num_nans > 0:
                    missing = True
                    if num_nans == y.size:
                        missing_only = True
                    else:
                        missing_only = False
                else:
                    missing = False
                    missing_only = False
                    
                # Split
                split_val_conditions = [y.size > 1,
                                        missing_only == False]
                if all(split_val_conditions):
                    split = DecisionTreeClassifier(criterion=self.criterion, splitter=self.splitter, 
                                                   max_depth=1, max_features=self.max_features, 
                                                   random_state=self.random_state)
                    if missing:
                        nans = np.isnan(X).any(axis=1)
                        split.fit(X[~nans], y[~nans])
                    else:
                        split.fit(X, y)
                else:
                    split = None
                
                # Node information
                node = {"index": container["index_node_global"],
                        "model": model_node,
                        "data": (X, y),
                        "classes_in": np.unique(y),
                        "num_classes": self.n_classes_in,
                        "split": split, 
                        "missing": {"missing": missing, "missing_only": missing_only},
                        "missing_side": None,
                        "children": {"left": None, "right": None},
                        "depth": depth}
                container["index_node_global"] += 1
                return node
            
            def _splitter(node):
                """Perform the split of a node."""
                # Extract data
                X, y = node["data"]
                depth = node["depth"]
                split = node["split"]
                missing = node["missing"]["missing"]
                missing_only = node["missing"]["missing_only"]
            
                did_split = False
                data = None
                
                # Perform split if the conditions are met
                stopping_criteria = [depth >= 0,
                                     depth < self.max_depth,
                                     np.unique(y).size > 1,
                                     missing_only == False]
            
                if all(stopping_criteria):
                    if missing:
                        nans = np.isnan(X).any(axis=1)
                        X_withoutnans, y_withoutnans = X[~nans], y[~nans]
                        leafs = split.apply(X_withoutnans)
                        (X_left, y_left), (X_right, y_right) = (np.squeeze(X_withoutnans[np.argwhere(leafs==1),:]), np.squeeze(y_withoutnans[np.argwhere(leafs==1)])), (np.squeeze(X_withoutnans[np.argwhere(leafs==2),:]), np.squeeze(y_withoutnans[np.argwhere(leafs==2)]))                    
                    else:
                        leafs = split.apply(X)
                        (X_left, y_left), (X_right, y_right) = (np.squeeze(X[np.argwhere(leafs==1),:]), np.squeeze(y[np.argwhere(leafs==1)])), (np.squeeze(X[np.argwhere(leafs==2),:]), np.squeeze(y[np.argwhere(leafs==2)]))                                                
                    
                    N_left, N_right = y_left.size, y_right.size

                    split_conditions = [N_left >= self.min_samples_leaf,
                                        N_right >= self.min_samples_leaf]
                    
                    if all(split_conditions):                        
                        did_split = True
                        
                        if N_left == 1:
                            X_left = X_left.reshape(-1, 1).T
                            node["missing_side"] = 'left'                            
                            if missing:    
                                X_left = np.append(X_left, X[nans], axis=0)
                                y_left = np.append([y_left], y[nans], axis=0)
                            
                        if N_right == 1:
                            X_right = X_right.reshape(-1, 1).T                            
                            if N_left > 1:
                                node["missing_side"] = 'right'
                                if missing:
                                    X_right = np.append(X_right, X[nans], axis=0)
                                    y_right = np.append([y_right], y[nans], axis=0)
                        
                        score_conditions = [N_left > 1,
                                            N_right > 1]
                        if all(score_conditions):
                            if split.score(X_left, y_left) > split.score(X_right, y_right):
                                node["missing_side"] = 'left'
                                if missing:                                
                                    X_left = np.append(X_left, X[nans], axis=0)
                                    y_left = np.append(y_left, y[nans], axis=0)
                            else:
                                node["missing_side"] = 'right'
                                if missing:
                                    X_right = np.append(X_right, X[nans], axis=0)
                                    y_right = np.append(y_right, y[nans], axis=0)
                            
                        data = [(X_left, y_left), (X_right, y_right)]
            
                result = {"did_split": did_split,
                          "data": data}
                return result

            def _split_traverse_node(node, container):
                """Process splitting results and continue with child nodes."""
                # Perform split and collect result
                result = _splitter(node)

                # Return terminal node if no split
                if not result["did_split"]:
                    if self.verbose > 0 and self.n_jobs == None:
                        depth_spacing_str = " ".join([" "] * node["depth"])
                        print(" {}*leaf {} @ depth {}: Unique_y {},  N_samples {}".format(depth_spacing_str, node["index"], node["depth"], np.unique(node["data"][1]), np.unique(node["data"][1], return_counts=True)[1]))
                    return
                del node["data"]

                # Extract splitting results
                (X_left, y_left), (X_right, y_right) = result["data"]

                # Report created node to user
                if self.verbose > 0 and self.n_jobs == None:
                    depth_spacing_str = " ".join([" "] * node["depth"])
                    print(" {}node {} @ depth {}: dataset={}, N_left={}, N_right={}".format(depth_spacing_str, node["index"], node["depth"], (X_left.shape[0]+X_right.shape[0], X_left.shape[1]), X_left.shape[0], X_right.shape[0]))

                # Create child nodes
                node["children"]["left"] = _create_node(X_left, y_left, node["depth"]+1, container)
                node["children"]["right"] = _create_node(X_right, y_right, node["depth"]+1, container)
                
                # Split nodes
                _split_traverse_node(node["children"]["left"], container)
                _split_traverse_node(node["children"]["right"], container)

            container = {"index_node_global": 0}
            if self.verbose > 0 and self.n_jobs == None:
                print('\nNew Tree')
            root = _create_node(X, y, 0, container)  
            _split_traverse_node(root, container) 
            return root        
        
        self.tree = _build_tree(X, y)
        return self

        
    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        
        def _predict(node, x):
            no_children = node["children"]["left"] is None and \
                          node["children"]["right"] is None
            if no_children:
                y_pred_x = node["model"].predict(x.reshape(-1, 1).T)[0]
                return node["classes_in"][y_pred_x]
            else:
                pred_proba = np.around(node["model"].predict_proba(x.reshape(-1, 1).T), 6)
                c = 0
                for i in range(0, node["num_classes"]):
                    x = np.insert(x.reshape(-1, 1).T, x.reshape(-1, 1).T.shape[1], 0, axis=1)
                    if i in node["classes_in"]:
                        if node["classes_in"].size == 1:
                            x[:,-1] = pred_proba[:,1]
                        else:
                            x[:,-1] = pred_proba[:,c]
                            c += 1
                if np.isnan(x).sum() > 0:
                    if node["missing_side"] == 'left':
                        x_left, x_right = x.reshape(-1, 1).T, []
                    else:
                        x_left, x_right = [], x.reshape(-1, 1).T
                else:   
                    leafs = node["split"].apply(x.reshape(-1, 1).T)
                    x_left, x_right = np.squeeze(x.reshape(-1, 1).T[np.argwhere(leafs==1),:]), np.squeeze(x.reshape(-1, 1).T[np.argwhere(leafs==2),:])
                if len(x_left) > 0:
                    return _predict(node["children"]["left"], x_left)
                else:
                    return _predict(node["children"]["right"], x_right)
        y_pred = np.array([_predict(self.tree, x) for x in X])
        return y_pred
    
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The class probabilities of the input samples. 
        """
        
        def _predict_proba(node, x):
            no_children = node["children"]["left"] is None and \
                          node["children"]["right"] is None
            if no_children:
                y_pred_x = np.around(node["model"].predict_proba(x.reshape(-1, 1).T), 6)
                d = 0
                for j in range(0, node["num_classes"]):
                    x = np.insert(x.reshape(-1, 1).T, x.reshape(-1, 1).T.shape[1], 0, axis=1)
                    if j in node["classes_in"]: 
                        if node["classes_in"].size == 1:
                            x[:,-1] = y_pred_x[:,1]
                        else:
                            x[:,-1] = y_pred_x[:,d]
                            d += 1
                return x[:, -node["num_classes"]:][0]
            else:
                pred_proba = np.around(node["model"].predict_proba(x.reshape(-1, 1).T), 6)
                c = 0
                for i in range(0, node["num_classes"]):
                    x = np.insert(x.reshape(-1, 1).T, x.reshape(-1, 1).T.shape[1], 0, axis=1)
                    if i in node["classes_in"]:
                        if node["classes_in"].size == 1:
                            x[:,-1] = pred_proba[:,1]
                        else:
                            x[:,-1] = pred_proba[:,c]
                            c += 1
                if np.isnan(x).sum() > 0:
                    if node["missing_side"] == 'left':
                        x_left, x_right = x.reshape(-1, 1).T, []
                    else:
                        x_left, x_right = [], x.reshape(-1, 1).T
                else:
                    leafs = node["split"].apply(x.reshape(-1, 1).T)
                    x_left, x_right = np.squeeze(x.reshape(-1, 1).T[np.argwhere(leafs==1),:]), np.squeeze(x.reshape(-1, 1).T[np.argwhere(leafs==2),:])
                if len(x_left) > 0:
                    return _predict_proba(node["children"]["left"], x_left)
                else:
                    return _predict_proba(node["children"]["right"], x_right)

        y_pred = np.array([_predict_proba(self.tree, x) for x in X])
        return y_pred
    
    
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
    
    
class LCETreeRegressor(RegressorMixin, BaseEstimator):
    """
    A LCE Tree regressor.
    

    Parameters
    ----------      
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
        range(1, xgb_max_n_estimators+xgb_n_estimators_step, xgb_n_estimators_step).

    xgb_max_depth : int, default= 10
        Maximum tree depth for XGBoost base learners. The range of XGBoost max_depth 
        for Hyperopt is: range(1, xgb_max_depth+1).
        
    xgb_min_learning_rate : float, default=0.05
        Minimum learning rate of XGBoost. The learning rate corresponds to the 
        step size shrinkage used in update to prevent overfitting. After each 
        boosting step, we can directly get the weights of new features, 
        and the learning rate shrinks the feature weights to make the boosting 
        process more conservative. 
        
    xgb_max_learning_rate : float, default=0.5
        Maximum learning rate of XGBoost.
    
    xgb_learning_rate_step : float, default=0.05
        Spacing between XGBoost learning_rate. The range of XGBoost learning_rate 
        for hyperparameter optimization (Hyperopt) is: 
        np.arange(xgb_min_learning_rate, xgb_max_learning_rate+xgb_learning_rate_step, xgb_learning_rate_step).
    
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
        np.arange(xgb_min_gamma, xgb_max_gamma+xgb_gamma_step, xgb_gamma_step).
    
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
        range(xgb_min_min_child_weight, xgb_max_min_child_weight+xgb_min_child_weight_step, xgb_min_child_weight_step).
        
    xgb_subsample : float, default=0.8 
        XGBoost subsample ratio of the training instances. Setting it to 0.5 means 
        that XGBoost would randomly sample half of the training data prior to 
        growing trees. and this will prevent overfitting. Subsampling will occur 
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
        np.arange(xgb_min_reg_alpha, xgb_max_reg_alpha+xgb_reg_alpha_step, xgb_reg_alpha_step).
                 
    xgb_min_reg_lambda : float, default=0.01
        Minimum reg_lambda of XGBoost. reg_lambda corresponds to the L2 regularization 
        term on the weights. Increasing this value will make XGBoost model more 
        conservative.
    
    xgb_max_reg_lambda : float, default=0.1
        Maximum reg_lambda of XGBoost.
    
    xgb_reg_lambda_step : float, default=0.05
        Spacing between XGBoost reg_lambda. The range of XGBoost reg_lambda for 
        hyperparameter optimization (Hyperopt) is: 
        np.arange(xgb_min_reg_lambda, xgb_max_reg_lambda+xgb_reg_lambda_step, xgb_reg_lambda_step).
    
    n_jobs : int, default=None
        The number of jobs to run in parallel. 
        ``None`` means 1. ``-1`` means using all processors. 

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the sampling of the features to consider when 
        looking for the best split at each node (if ``max_features < n_features``), 
        the base classifier (XGBoost) and the Hyperopt algorithm.
 
    verbose : int, default=0
        Controls the verbosity when fitting.
        
    Attributes
    ----------
    n_features_in_ : int
        The number of features when ``fit`` is performed.
    """

    def __init__(self, criterion='squared_error', splitter='best', max_depth=2, 
                 max_features=None, min_samples_leaf=5, n_iter=10, metric = 'neg_mean_squared_error',
                 xgb_max_n_estimators=100, xgb_n_estimators_step=10, xgb_max_depth=10,
                 xgb_min_learning_rate=0.05, xgb_max_learning_rate=0.5, xgb_learning_rate_step=0.05, 
                 xgb_booster='gbtree', xgb_min_gamma=0.05, xgb_max_gamma=0.5, xgb_gamma_step=0.05,
                 xgb_min_min_child_weight=3, xgb_max_min_child_weight=10, xgb_min_child_weight_step=1, 
                 xgb_subsample=0.8, xgb_colsample_bytree=0.8,
                 xgb_colsample_bylevel=1.0, xgb_colsample_bynode=1.0, 
                 xgb_min_reg_alpha=0.01, xgb_max_reg_alpha=0.1, xgb_reg_alpha_step=0.05, 
                 xgb_min_reg_lambda=0.01, xgb_max_reg_lambda=0.1, xgb_reg_lambda_step=0.05,
                 n_jobs=None, random_state=None, verbose=0):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.max_features = max_features
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
        
    
    def fit(self, X, y):
        """
        Build a LCE tree from the training set (X, y).

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
        self.n_features_in_ = X.shape[1]        
        
        def _build_tree(X, y):
            """Build a LCE tree."""
            global index_node_global

            def _create_node(X, y, depth, container):
                """Create a node in the tree."""
                # Add XGBoost predictions as features to the dataset
                model_node = xgb_opt_regressor(X, y, n_iter=self.n_iter,
                                               metric = self.metric,
                                               n_estimators=self.xgb_max_n_estimators,
                                               n_estimators_step = self.xgb_n_estimators_step,
                                               max_depth=self.xgb_max_depth,
                                               min_learning_rate = self.xgb_min_learning_rate,
                                               max_learning_rate = self.xgb_max_learning_rate,
                                               learning_rate_step = self.xgb_learning_rate_step,
                                               booster = self.xgb_booster,
                                               min_gamma = self.xgb_min_gamma,
                                               max_gamma = self.xgb_max_gamma,
                                               gamma_step = self.xgb_gamma_step,
                                               min_min_child_weight = self.xgb_min_min_child_weight,
                                               max_min_child_weight = self.xgb_max_min_child_weight,
                                               min_child_weight_step = self.xgb_min_child_weight_step,
                                               subsample=self.xgb_subsample, 
                                               colsample_bytree = self.xgb_colsample_bytree,
                                               colsample_bylevel = self.xgb_colsample_bylevel,
                                               colsample_bynode = self.xgb_colsample_bynode,
                                               min_reg_alpha = self.xgb_min_reg_alpha,
                                               max_reg_alpha = self.xgb_max_reg_alpha,
                                               reg_alpha_step = self.xgb_reg_alpha_step,
                                               min_reg_lambda  = self.xgb_min_reg_lambda,
                                               max_reg_lambda  = self.xgb_max_reg_lambda,
                                               reg_lambda_step = self.xgb_reg_lambda_step,
                                               random_state=self.random_state)
                preds = np.around(model_node.predict(X), 6)
                X = np.insert(X, X.shape[1], 0, axis=1)
                X[:,-1] = preds
                
                # Missing data information
                num_nans = np.isnan(X).any(axis=1).sum()
                if num_nans > 0:
                    missing = True
                    if num_nans == y.size:
                        missing_only = True
                    else:
                        missing_only = False
                else:
                    missing = False
                    missing_only = False
                    
                # Split
                split_val_conditions = [y.size > 1,
                                        missing_only == False]
                if all(split_val_conditions):
                    split = DecisionTreeRegressor(criterion=self.criterion, splitter=self.splitter, 
                                                  max_depth=1, max_features=self.max_features, 
                                                  random_state=self.random_state)
                    if missing:
                        nans = np.isnan(X).any(axis=1)
                        split.fit(X[~nans], y[~nans])
                    else:
                        split.fit(X, y)
                else:
                    split = None
                
                # Node information
                node = {"index": container["index_node_global"],
                        "model": model_node,
                        "data": (X, y),
                        "split": split, 
                        "missing": {"missing": missing, "missing_only": missing_only},
                        "missing_side": None,
                        "children": {"left": None, "right": None},
                        "depth": depth}
                container["index_node_global"] += 1
                return node
            
            def _splitter(node):
                """Perform the split of a node."""
                # Extract data
                X, y = node["data"]
                depth = node["depth"]
                split = node["split"]
                missing = node["missing"]["missing"]
                missing_only = node["missing"]["missing_only"]
            
                did_split = False
                data = None
                
                # Perform split if the conditions are met
                stopping_criteria = [depth >= 0,
                                     depth < self.max_depth,
                                     X[:,0].size > 1,
                                     missing_only == False]
            
                if all(stopping_criteria):
                    if missing:
                        nans = np.isnan(X).any(axis=1)
                        X_withoutnans, y_withoutnans = X[~nans], y[~nans]
                        leafs = split.apply(X_withoutnans)
                        (X_left, y_left), (X_right, y_right) = (np.squeeze(X_withoutnans[np.argwhere(leafs==1),:]), np.squeeze(y_withoutnans[np.argwhere(leafs==1)])), (np.squeeze(X_withoutnans[np.argwhere(leafs==2),:]), np.squeeze(y_withoutnans[np.argwhere(leafs==2)]))                    
                    else:
                        leafs = split.apply(X)
                        (X_left, y_left), (X_right, y_right) = (np.squeeze(X[np.argwhere(leafs==1),:]), np.squeeze(y[np.argwhere(leafs==1)])), (np.squeeze(X[np.argwhere(leafs==2),:]), np.squeeze(y[np.argwhere(leafs==2)]))                                                
                    
                    N_left, N_right = y_left.size, y_right.size

                    split_conditions = [N_left >= self.min_samples_leaf,
                                        N_right >= self.min_samples_leaf]
                    
                    if all(split_conditions):                        
                        did_split = True
                        
                        if N_left == 1:
                            X_left = X_left.reshape(-1, 1).T
                            node["missing_side"] = 'left'                            
                            if missing:    
                                X_left = np.append(X_left, X[nans], axis=0)
                                y_left = np.append([y_left], y[nans], axis=0)
                            
                        if N_right == 1:
                            X_right = X_right.reshape(-1, 1).T                            
                            if N_left > 1:
                                node["missing_side"] = 'right'
                                if missing:
                                    X_right = np.append(X_right, X[nans], axis=0)
                                    y_right = np.append([y_right], y[nans], axis=0)
                        
                        score_conditions = [N_left > 1,
                                            N_right > 1]
                        if all(score_conditions):
                            if split.score(X_left, y_left) > split.score(X_right, y_right):
                                node["missing_side"] = 'left'
                                if missing:                                
                                    X_left = np.append(X_left, X[nans], axis=0)
                                    y_left = np.append(y_left, y[nans], axis=0)
                            else:
                                node["missing_side"] = 'right'
                                if missing:
                                    X_right = np.append(X_right, X[nans], axis=0)
                                    y_right = np.append(y_right, y[nans], axis=0)
                            
                        data = [(X_left, y_left), (X_right, y_right)]
            
                result = {"did_split": did_split,
                          "data": data}
                return result

            def _split_traverse_node(node, container):
                """Process splitting results and continue with child nodes."""
                # Perform split and collect result
                result = _splitter(node)

                # Return terminal node if no split
                if not result["did_split"]:
                    if self.verbose > 0 and self.n_jobs == None:
                        depth_spacing_str = " ".join([" "] * node["depth"])
                        print(" {}*leaf {} @ depth {}: Unique_y {},  N_samples {}".format(depth_spacing_str, node["index"], node["depth"], np.unique(node["data"][1]), np.unique(node["data"][1], return_counts=True)[1]))
                    return
                del node["data"]

                # Extract splitting results
                (X_left, y_left), (X_right, y_right) = result["data"]

                # Report created node to user
                if self.verbose > 0 and self.n_jobs == None:
                    depth_spacing_str = " ".join([" "] * node["depth"])
                    print(" {}node {} @ depth {}: dataset={}, N_left={}, N_right={}".format(depth_spacing_str, node["index"], node["depth"], (X_left.shape[0]+X_right.shape[0], X_left.shape[1]), X_left.shape[0], X_right.shape[0]))

                # Create child nodes
                node["children"]["left"] = _create_node(X_left, y_left, node["depth"]+1, container)
                node["children"]["right"] = _create_node(X_right, y_right, node["depth"]+1, container)
                
                # Split nodes
                _split_traverse_node(node["children"]["left"], container)
                _split_traverse_node(node["children"]["right"], container)

            container = {"index_node_global": 0}
            if self.verbose > 0 and self.n_jobs == None:
                print('\nNew Tree')
            root = _create_node(X, y, 0, container)  
            _split_traverse_node(root, container) 
            return root        
        
        self.tree = _build_tree(X, y)
        return self

        
    def predict(self, X):
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        
        def _predict(node, x):
            no_children = node["children"]["left"] is None and \
                          node["children"]["right"] is None
            if no_children:
                y_pred_x = node["model"].predict(x.reshape(-1, 1).T)[0]
                return y_pred_x
            else:
                preds = np.around(node["model"].predict(x.reshape(-1, 1).T), 6)
                x = np.insert(x.reshape(-1, 1).T, x.reshape(-1, 1).T.shape[1], 0, axis=1)
                x[:,-1] = preds
                if np.isnan(x).sum() > 0:
                    if node["missing_side"] == 'left':
                        x_left, x_right = x.reshape(-1, 1).T, []
                    else:
                        x_left, x_right = [], x.reshape(-1, 1).T
                else:
                    leafs = node["split"].apply(x.reshape(-1, 1).T)
                    x_left, x_right = np.squeeze(x.reshape(-1, 1).T[np.argwhere(leafs==1),:]), np.squeeze(x.reshape(-1, 1).T[np.argwhere(leafs==2),:])
                if len(x_left) > 0:
                    return _predict(node["children"]["left"], x_left)
                else:
                    return _predict(node["children"]["right"], x_right)
        y_pred = np.array([_predict(self.tree, x) for x in X])
        return y_pred
    
    
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