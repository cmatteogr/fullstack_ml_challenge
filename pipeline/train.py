"""
"""

import numpy as np
from xgboost import XGBRFRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os


def train(X_train, y_train, results_folder_path: str) -> tuple:
    """
    """
    print("start training")

    # Define XGBoost model
    xgb_model = XGBRFRegressor(objective='reg:squarederror', device="cuda", eval_metric='rmse')
    # define parameter ranges for tuning
    param_grid = {
        'learning_rate': Real(1e-3, 3, prior='log-uniform'),
        'max_depth': Integer(3, 30),
        # NOTE: in run time a warning appears because max_depth applies only to gbtree and dart boosters, as they use decision trees
        'n_estimators': Integer(5, 500, prior='log-uniform'),
        'booster': Categorical(['gbtree', 'gblinear', 'dart']),
    }
    # bayesian cv models
    opt = BayesSearchCV(xgb_model, param_grid, n_iter=50, cv=5, random_state=0, verbose=2)
    # executes bayesian optimization
    _ = opt.fit(X_train, y_train)
    print(opt.score(X_train, y_train))

    # Best model from grid search
    best_model = opt.best_estimator_

    # Predict on test set
    y_pred = best_model.predict(X_train)

    # Evaluating model performance
    print("conditions model performance")
    mse = mean_squared_error(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    rmse = np.sqrt(mse)

    print("export results")
    # Built result JSON train report
    results_dict = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }
    print(results_dict)

    # save model
    model_filepath = os.path.join(results_folder_path, 'xgb_price_regression.json')
    best_model.save_model(model_filepath)

    print("training completed")

    # return model filepath
    return results_dict, model_filepath

