"""
"""

import numpy as np
from xgboost import XGBRFRegressor
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import optuna
from sklearn.model_selection import cross_val_score
from utils.constants import REGRESSION_MODEL_FILEPATH, ARTIFACTS_FOLDER


def train(X_train, y_train, results_folder_path: str) -> tuple:
    """
    """
    print("start training")

    def objective(trial):
        param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'device': "cuda",
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 30),  # Corrected range
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, log=True),
            'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
        }
        # init XGBoost model
        xgb_model = XGBRFRegressor(**param)
        # use cross validation to avoid overfitting
        cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        mse_cv = -np.mean(cv_scores)
        # return mse
        return mse_cv

    # use optuna to find the best hyperparameters
    # save the study to have checkpoints
    # NOTE: we can set any prune and optimization method like Bayes
    study_name = "price_regression_v3"
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True,
                                direction='minimize')
    # optimize the objective
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # get final best parameters
    final_model_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'device': "cuda",
        **study.best_trial.params  # Unpack the tuned parameters
    }
    # fit model with
    best_model = XGBRFRegressor(**final_model_params)
    best_model.fit(X_train, y_train)

    # Predict on test set
    y_pred = best_model.predict(X_train)

    # eEvaluating model performance
    print("conditions model performance")
    mse = mean_squared_error(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    rmse = np.sqrt(mse)

    print("export results")
    # built result JSON train report
    results_dict = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }
    print(results_dict)
    # Save JSON results report
    results_json_filepath = os.path.join(results_folder_path, 'train_result.json')
    with open(results_json_filepath, 'w') as f:
        json.dump(results_dict, f, indent=4)

    # save model
    model_filepath = os.path.join(ARTIFACTS_FOLDER, REGRESSION_MODEL_FILEPATH)
    best_model.save_model(model_filepath)

    print("training completed")

    # return model filepath
    return results_dict, model_filepath
