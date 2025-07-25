"""
Evaluation step
"""
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import json


def evaluation(X_test, y_test, model_filepath: str, results_folder_path:str) -> dict:
    """
    """
    print('start evaluation')

    # Load the model
    loaded_model = xgb.XGBRFRegressor()
    loaded_model.load_model(model_filepath)

    # Use the loaded model
    y_pred = loaded_model.predict(X_test)

    # Evaluating model performance
    print("conditions model performance")
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
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
    # Save JSON results report
    results_json_filepath = os.path.join(results_folder_path, 'evaluation_result.json')
    with open(results_json_filepath, 'w') as f:
        json.dump(results_dict, f, indent=4)

    print("evaluation completed")


    return results_dict
