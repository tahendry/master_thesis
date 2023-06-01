"""
Date: 09.04.2023
Author: Reto Hendry

"""

import numpy as np
import pandas as pd
import datetime
import os
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score
from pycaret.classification import (
    setup, compare_models, tune_model, save_model,
    predict_model, get_metrics, pull, evaluate_model)

os.environ['PYDEVD_WARN_EVALUATION_TIMEOUT'] = '10.0'
os.environ['PYDEVD_UNBLOCK_THREADS_TIMEOUT'] = '5.0'

def run_pycaret(sample_array_4d, df_label, feature_list, component, resample_cube, randomize_labels=False):
    """
    Run pycaret on the input data.

    Parameters
    ----------
    sample_array_4d : numpy array
        4d array with [sample, x, y, z].
    df_label : pandas dataframe
        df_label with ground truth in column "Cond".
    feature_list : list
        a list of the features to be used (elements in flattened array)
    component : int
        The component number
    resample_cube : int
        the size of the cube to reshape the array 5 -> 5x5x5
    randomize_labels : bool, optional (default=False)
        if True, the labels are mixed randomly
        This is to check how the results change.


    Returns
    -------
    results : pandas dataframe
        a dataframe with the results of the best model of the pycaret run

    """
    
    if randomize_labels:
        # shuffle the labels
        df_label["Cond"] = np.random.permutation(df_label["Cond"].values)

    # Define the cross-validation strategy
    cv_stratified = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)

    # reshape 4d array to dataframe
    sample_df = pd.DataFrame(
        sample_array_4d.reshape(sample_array_4d.shape[0], -1)
    ).iloc[:, feature_list]

    # Combine sample_df and df_label
    data = pd.concat([sample_df, df_label["Cond"]], axis=1)

    # split data into train and test
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["Cond"]
    )

    # Setting up the pycaret environment
    exp = setup(
        data = train_data,  # training data
        target = "Cond",  # target variable
        test_data = test_data,  # hold out data for testing
        keep_features = list(data.columns[:-1]),  # keep all features (cols)
        normalize = True,  # normalize data
        normalize_method = "zscore",  #  z = (x - u) / s
        fold_strategy = cv_stratified,  # predefined CV object
        n_jobs = -1,  # use all processors
        session_id = 1,  # random state
        verbose = True,  # print information grid
    )

    # Compare and select the best model
    best_model = compare_models()
    # Tune the best model
    tuned_model, tuner = tune_model(best_model, optimize="Accuracy", 
                                    n_iter = 100, return_tuner=True)

    # get metrics of the tuned model
    metrics_tuned_model = pull(tuned_model)

    # Save best model
    id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model(tuned_model, f"./param_sweep_best_models/pycaret_best_models/pycaret_{id}")
    """
    sometimes, the model tuned, is still not better than the best model
    in this case, the best model is saved, respectively the tuned model
    results in the same model
    """

    # get cross validation results
    cv_mean_accuracy = metrics_tuned_model.loc["Mean", "Accuracy"]
    cv_std = metrics_tuned_model.loc["Std", "Accuracy"]

    # Make predictions on the test set
    predictions = predict_model(tuned_model)  # with hold out data (test_data)

    # Calculate the accuracy and F1 score on the test set
    test_accuracy = accuracy_score(
        test_data["Cond"], predictions["prediction_label"]
    )
    test_f1_score = f1_score(
        test_data["Cond"], predictions["prediction_label"], average="weighted"
    )
    test_precision_score = precision_score(
        test_data["Cond"], predictions["prediction_label"], average="weighted"
    )

    # Create a DataFrame to store the results
    results = pd.DataFrame({
        "id": [f"pycaret_{id}"],
        "component": [component],
        "resample_cube": [resample_cube],
        "number_of_features": [len(feature_list)],
        "CV_Accuracy_(Training)": [np.round(cv_mean_accuracy, 4)],
        "CV_Std_(Training)": [np.round(cv_std, 4)],
        "Accuracy_(Test)": [np.round(test_accuracy, 4)],
        "F1_Score_(Test)": [np.round(test_f1_score, 4)],
        "Precision_Score_(Test)": [np.round(test_precision_score, 4)],
    })

    return results
