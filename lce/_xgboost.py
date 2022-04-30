import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import check_scoring
import xgboost as xgb


def xgb_opt_classifier(X, y, n_iter=10, metric='accuracy', n_estimators=100, 
                       n_estimators_step=10, max_depth=10, min_learning_rate=0.05, 
                       max_learning_rate=0.5,learning_rate_step=0.05, booster='gbtree',
                       min_gamma=0.05, max_gamma=0.5, gamma_step=0.05, 
                       min_min_child_weight=3, max_min_child_weight=10, min_child_weight_step=1, 
                       subsample=0.8, colsample_bytree=0.8, colsample_bylevel=1.0, 
                       colsample_bynode=1.0, min_reg_alpha=0.01, max_reg_alpha=0.1, 
                       reg_alpha_step=0.05, min_reg_lambda=0.01, max_reg_lambda=0.1, 
                       reg_lambda_step=0.05, random_state=None):
    """
    Get XGBoost model with the best hyperparameters configuration.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The training input samples.

    y : array-like of shape (n_samples,)
        The class labels.
        
    n_iter: int, default=10
        Number of iterations to set the hyperparameters of the base classifier (XGBoost)
        in Hyperopt.
    
    metric: string, default="accuracy"
        The score of the base classifier (XGBoost) optimized by Hyperopt. Supported metrics 
        are the ones from `scikit-learn <https://scikit-learn.org/stable/modules/model_evaluation.html>`_.
        
    n_estimators : int, default=100
        The maximum number of XGBoost estimators. The number of estimators of 
        XGBoost corresponds to the number of boosting rounds.
        
    n_estimators_step : int, default=10
        Spacing between XGBoost n_estimators.

    max_depth : int, default= 10
        Maximum tree depth for XGBoost base learners.
        
    min_learning_rate : float, default=0.05
        Minimum learning rate of XGBoost. The learning rate corresponds to the 
        step size shrinkage used in update to prevent overfitting. After each 
        boosting step, we can directly get the weights of new features, 
        and the learning rate shrinks the feature weights to make the boosting 
        process more conservative. 
        
    max_learning_rate : float, default=0.5
        Maximum learning rate of XGBoost.
    
    learning_rate_step : float, default=0.05
        Spacing between XGBoost learning_rate.
    
    booster : {"dart", "gblinear", "gbtree"}, default="gbtree"
        The type of booster to use. "gbtree" and "dart" use tree based models 
        while "gblinear" uses linear functions.
        
    min_gamma : float, default=0.05
        Minimum gamma of XGBoost. Gamma corresponds to the minimum loss reduction 
        required to make a further partition on a leaf node of the tree. 
        The larger gamma is, the more conservative XGBoost algorithm will be.
    
    max_gamma : float, default=0.5 
        Maximum gamma of XGBoost.
    
    gamma_step : float, default=0.05,
        Spacing between XGBoost gamma.
    
    min_min_child_weight : int, default=3 
        Minimum min_child_weight of XGBoost. min_child_weight defines the
        minimum sum of instance weight (hessian) needed in a child. If the tree 
        partition step results in a leaf node with the sum of instance weight 
        less than min_child_weight, then the building process will give up further 
        partitioning. The larger min_child_weight is, the more conservative XGBoost 
        algorithm will be.
    
    max_min_child_weight : int, default=10
        Minimum min_child_weight of XGBoost.
    
    min_child_weight_step : int, default=1,
        Spacing between XGBoost min_child_weight.
        
    subsample : float, default=0.8 
        XGBoost subsample ratio of the training instances. Setting it to 0.5 means 
        that XGBoost would randomly sample half of the training data prior to 
        growing trees. and this will prevent overfitting. Subsampling will occur 
        once in every boosting iteration.
    
    colsample_bytree : float, default=0.8
        XGBoost subsample ratio of columns when constructing each tree. 
        Subsampling occurs once for every tree constructed.
    
    colsample_bylevel : float, default=1.0
        XGBoost subsample ratio of columns for each level. Subsampling occurs 
        once for every new depth level reached in a tree. Columns are subsampled 
        from the set of columns chosen for the current tree.
    
    colsample_bynode : float, default=1.0
        XGBoost subsample ratio of columns for each node (split). Subsampling 
        occurs once every time a new split is evaluated. Columns are subsampled 
        from the set of columns chosen for the current level.
        
    min_reg_alpha : float, default=0.01
        Minimum reg_alpha of XGBoost. reg_alpha corresponds to the L1 regularization 
        term on the weights. Increasing this value will make XGBoost model more 
        conservative.
    
    max_reg_alpha : float, default=0.1
        Maximum reg_alpha of XGBoost.
    
    reg_alpha_step : float, default=0.05
        Spacing between XGBoost reg_alpha.
                 
    min_reg_lambda : float, default=0.01
        Minimum reg_lambda of XGBoost. reg_lambda corresponds to the L2 regularization 
        term on the weights. Increasing this value will make XGBoost model more 
        conservative.
    
    max_reg_lambda : float, default=0.1
        Maximum reg_lambda of XGBoost.
    
    reg_lambda_step : float, default=0.05
        Spacing between XGBoost reg_lambda.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the base classifier XGBoost and
        the Hyperopt algorithm.

    Returns
    -------
    model: object
        XGBoost model with the best configuration and fitted on the input data.
    """
    # Parameters
    classes, y = np.unique(y, return_inverse=True)
    n_classes = classes.size
    n_estimators = range(1, n_estimators+n_estimators_step, n_estimators_step)
    max_depth = range(1, max_depth+1)
    learning_rate = np.arange(min_learning_rate, max_learning_rate+learning_rate_step, 
                              learning_rate_step)
    gamma = np.arange(min_gamma, max_gamma+gamma_step, 
                              gamma_step)
    min_child_weight = range(min_min_child_weight, max_min_child_weight+min_child_weight_step, min_child_weight_step)
    reg_alpha = np.arange(min_reg_alpha, max_reg_alpha+reg_alpha_step, 
                              reg_alpha_step)
    reg_lambda = np.arange(min_reg_lambda, max_reg_lambda+reg_lambda_step, 
                              reg_lambda_step)
    subsample = [subsample]
    colsample_bytree = [colsample_bytree]
    colsample_bylevel = [colsample_bylevel]
    colsample_bynode = [colsample_bynode]
    
    space = {
        'n_estimators': hp.choice('n_estimators', n_estimators),
        'max_depth': hp.choice('max_depth', max_depth),
        'learning_rate': hp.choice('learning_rate', learning_rate),
        'booster':booster,
        'gamma':hp.choice('gamma',gamma),
        'min_child_weight': hp.choice('min_child_weight', min_child_weight),
        'subsample': hp.choice('subsample', subsample),
        'colsample_bytree': hp.choice('colsample_bytree', colsample_bytree),
        'colsample_bylevel': hp.choice('colsample_bylevel', colsample_bylevel),
        'colsample_bynode': hp.choice('colsample_bynode', colsample_bynode),
        'reg_alpha':hp.choice('reg_alpha',reg_alpha),
        'reg_lambda':hp.choice('reg_lambda',reg_lambda),
        'objective':'multi:softprob',
        'num_class':n_classes,
        'n_jobs':-1,
        'random_state':random_state,
    }
    
    # Get best configuration
    def p_model(params):
        clf = xgb.XGBClassifier(**params, use_label_encoder=False, verbosity=0)
        clf.fit(X, y)
        scorer = check_scoring(clf, scoring=metric)
        return scorer(clf, X, y)
    
    global best
    global best_print
    best=0
    best_print=0
    def f(params):
        global best
        global best_print
        perf = p_model(params)
        if perf > best:
            best = perf
        best_print=best
        return {'loss': -best, 'status': STATUS_OK}
    
    rstate = np.random.default_rng(random_state)
    best = fmin(
        fn=f,
        space=space,
        algo=tpe.suggest,
        max_evals=n_iter,
        trials=Trials(),
        rstate=rstate,
        verbose=0
    )
    
    # Fit best model
    final_params = {
        'n_estimators':n_estimators[best['n_estimators']],
        'max_depth':max_depth[best['max_depth']],
        'learning_rate':learning_rate[best['learning_rate']],
        'booster':booster,
        'gamma':gamma[best['gamma']],        
        'min_child_weight':min_child_weight[best['min_child_weight']],
        'subsample':subsample[best['subsample']],
        'colsample_bytree':colsample_bytree[best['colsample_bytree']],
        'colsample_bylevel':colsample_bylevel[best['colsample_bylevel']],
        'colsample_bynode':colsample_bynode[best['colsample_bynode']],
        'reg_alpha':reg_alpha[best['reg_alpha']],
        'reg_lambda':reg_lambda[best['reg_lambda']],
        'objective':'multi:softprob',
        'num_class':n_classes,
        'n_jobs':-1,
        'random_state':random_state,
    }
    clf = xgb.XGBClassifier(**final_params, use_label_encoder=False, verbosity=0)
    return clf.fit(X, y)


def xgb_opt_regressor(X, y, n_iter=10, metric='neg_mean_squared_error', n_estimators=100, 
                      n_estimators_step=10, max_depth=10, min_learning_rate=0.05, 
                      max_learning_rate=0.5, learning_rate_step=0.05, booster='gbtree',
                      min_gamma=0.05, max_gamma=0.5, gamma_step=0.05, min_min_child_weight=3, 
                      max_min_child_weight=10, min_child_weight_step=1, subsample=0.8, 
                      colsample_bytree=0.8, colsample_bylevel=1.0, colsample_bynode=1.0, 
                      min_reg_alpha=0.01, max_reg_alpha=0.1, reg_alpha_step=0.05, 
                      min_reg_lambda=0.01, max_reg_lambda=0.1, reg_lambda_step=0.05,
                      random_state=None):
    """
    Get XGBoost model with the best hyperparameters configuration.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The training input samples.

    y : array-like of shape (n_samples,)
        The target values (real numbers).
        
    n_iter: int, default=10
        Number of iterations to set the hyperparameters of the base regressor (XGBoost) 
        in Hyperopt.
        
    metric: string, default="neg_mean_squared_error"
        The score of the base regressor (XGBoost) optimized by Hyperopt. Supported metrics 
        are the ones from `scikit-learn <https://scikit-learn.org/stable/modules/model_evaluation.html>`_.
        
    n_estimators : int, default=100
        The maximum number of XGBoost estimators. The number of estimators of 
        XGBoost corresponds to the number of boosting rounds.
        
    n_estimators_step : int, default=10
        Spacing between XGBoost n_estimators.

    max_depth : int, default= 10
        Maximum tree depth for XGBoost base learners.
        
    min_learning_rate : float, default=0.05
        Minimum learning rate of XGBoost. The learning rate corresponds to the 
        step size shrinkage used in update to prevent overfitting. After each 
        boosting step, we can directly get the weights of new features, 
        and the learning rate shrinks the feature weights to make the boosting 
        process more conservative. 
        
    max_learning_rate : float, default=0.5
        Maximum learning rate of XGBoost.
    
    learning_rate_step : float, default=0.05
        Spacing between XGBoost learning_rate.
    
    booster : {"dart", "gblinear", "gbtree"}, default="gbtree"
        The type of booster to use. "gbtree" and "dart" use tree based models 
        while "gblinear" uses linear functions.
        
    min_gamma : float, default=0.05
        Minimum gamma of XGBoost. Gamma corresponds to the minimum loss reduction 
        required to make a further partition on a leaf node of the tree. 
        The larger gamma is, the more conservative XGBoost algorithm will be.
    
    max_gamma : float, default=0.5 
        Maximum gamma of XGBoost.
    
    gamma_step : float, default=0.05,
        Spacing between XGBoost gamma.
    
    min_min_child_weight : int, default=3 
        Minimum min_child_weight of XGBoost. min_child_weight defines the
        minimum sum of instance weight (hessian) needed in a child. If the tree 
        partition step results in a leaf node with the sum of instance weight 
        less than min_child_weight, then the building process will give up further 
        partitioning. The larger min_child_weight is, the more conservative XGBoost 
        algorithm will be.
    
    max_min_child_weight : int, default=10
        Minimum min_child_weight of XGBoost.
    
    min_child_weight_step : int, default=1,
        Spacing between XGBoost min_child_weight.
        
    subsample : float, default=0.8 
        XGBoost subsample ratio of the training instances. Setting it to 0.5 means 
        that XGBoost would randomly sample half of the training data prior to 
        growing trees. and this will prevent overfitting. Subsampling will occur 
        once in every boosting iteration.
    
    colsample_bytree : float, default=0.8
        XGBoost subsample ratio of columns when constructing each tree. 
        Subsampling occurs once for every tree constructed.
    
    colsample_bylevel : float, default=1.0
        XGBoost subsample ratio of columns for each level. Subsampling occurs 
        once for every new depth level reached in a tree. Columns are subsampled 
        from the set of columns chosen for the current tree.
    
    colsample_bynode : float, default=1.0
        XGBoost subsample ratio of columns for each node (split). Subsampling 
        occurs once every time a new split is evaluated. Columns are subsampled 
        from the set of columns chosen for the current level.
        
    min_reg_alpha : float, default=0.01
        Minimum reg_alpha of XGBoost. reg_alpha corresponds to the L1 regularization 
        term on the weights. Increasing this value will make XGBoost model more 
        conservative.
    
    max_reg_alpha : float, default=0.1
        Maximum reg_alpha of XGBoost.
    
    reg_alpha_step : float, default=0.05
        Spacing between XGBoost reg_alpha.
                 
    min_reg_lambda : float, default=0.01
        Minimum reg_lambda of XGBoost. reg_lambda corresponds to the L2 regularization 
        term on the weights. Increasing this value will make XGBoost model more 
        conservative.
    
    max_reg_lambda : float, default=0.1
        Maximum reg_lambda of XGBoost.
    
    reg_lambda_step : float, default=0.05
        Spacing between XGBoost reg_lambda.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the base classifier XGBoost and
        the Hyperopt algorithm.

    Returns
    -------
    model: object
        XGBoost model with the best configuration and fitted on the input data.
    """
    # Parameters
    n_estimators = range(1, n_estimators+n_estimators_step, n_estimators_step)
    max_depth = range(1, max_depth+1)
    learning_rate = np.arange(min_learning_rate, max_learning_rate+learning_rate_step, 
                              learning_rate_step)
    gamma = np.arange(min_gamma, max_gamma+gamma_step, 
                              gamma_step)
    min_child_weight = range(min_min_child_weight, max_min_child_weight+min_child_weight_step, min_child_weight_step)
    reg_alpha = np.arange(min_reg_alpha, max_reg_alpha+reg_alpha_step, 
                              reg_alpha_step)
    reg_lambda = np.arange(min_reg_lambda, max_reg_lambda+reg_lambda_step, 
                              reg_lambda_step)
    subsample = [subsample]
    colsample_bytree = [colsample_bytree]
    colsample_bylevel = [colsample_bylevel]
    colsample_bynode = [colsample_bynode]
    
    space = {
        'n_estimators': hp.choice('n_estimators', n_estimators),
        'max_depth': hp.choice('max_depth', max_depth),
        'learning_rate': hp.choice('learning_rate', learning_rate),
        'booster':booster,
        'gamma':hp.choice('gamma',gamma),
        'min_child_weight': hp.choice('min_child_weight', min_child_weight),
        'subsample': hp.choice('subsample', subsample),
        'colsample_bytree': hp.choice('colsample_bytree', colsample_bytree),
        'colsample_bylevel': hp.choice('colsample_bylevel', colsample_bylevel),
        'colsample_bynode': hp.choice('colsample_bynode', colsample_bynode),
        'reg_alpha':hp.choice('reg_alpha',reg_alpha),
        'reg_lambda':hp.choice('reg_lambda',reg_lambda),
        'objective':'reg:squarederror',
        'n_jobs':-1,
        'random_state':random_state,
    }
    
    # Get best configuration
    def p_model(params):
        reg = xgb.XGBRegressor(**params, verbosity=0)
        reg.fit(X, y)
        scorer = check_scoring(reg, scoring=metric)
        return scorer(reg, X, y)
    
    global best
    global best_print
    best=0
    best_print=0
    def f(params):
        global best
        global best_print
        perf = p_model(params)
        if perf > best:
            best = perf
        best_print=best
        return {'loss': best, 'status': STATUS_OK}
    
    rstate = np.random.default_rng(random_state)
    best = fmin(
        fn=f,
        space=space,
        algo=tpe.suggest,
        max_evals=n_iter,
        trials=Trials(),
        rstate=rstate,
        verbose=0
    )
    
    # Fit best model
    final_params = {
        'n_estimators':n_estimators[best['n_estimators']],
        'max_depth':max_depth[best['max_depth']],
        'learning_rate':learning_rate[best['learning_rate']],
        'booster':booster,
        'gamma':gamma[best['gamma']],        
        'min_child_weight':min_child_weight[best['min_child_weight']],
        'subsample':subsample[best['subsample']],
        'colsample_bytree':colsample_bytree[best['colsample_bytree']],
        'colsample_bylevel':colsample_bylevel[best['colsample_bylevel']],
        'colsample_bynode':colsample_bynode[best['colsample_bynode']],
        'reg_alpha':reg_alpha[best['reg_alpha']],
        'reg_lambda':reg_lambda[best['reg_lambda']],
        'objective':'reg:squarederror',
        'n_jobs':-1,
        'random_state':random_state,
    }
    reg = xgb.XGBRegressor(**final_params, verbosity=0)
    return reg.fit(X, y)