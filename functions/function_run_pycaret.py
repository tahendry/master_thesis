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

def run_pycaret(sample_array_4d, df_label, feature_list, component, reshape_cube):
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
    reshape_cube : int
        the size of the cube to reshape the array 5 -> 5x5x5

    Returns
    -------
    results : pandas dataframe
        a dataframe with the results of the best model of the pycaret run

    """

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
        # use_gpu = True,  # use GPU for some models if available
        session_id = 1,  # random state
        verbose = True,  # print information grid
    )

    # Compare and select the best model
    best_model = compare_models()
    # Tune the best model
    tuned_model = tune_model(best_model, optimize="Accuracy", n_iter = 50)

    # get metrics of the tuned model
    metrics_tuned_model = pull(tuned_model)

    # Save best model
    id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_model(tuned_model, f"pycaret_best_models/pycaret_{id}")

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
        "reshape_cube": [reshape_cube],
        "number of features": [len(feature_list)],
        "CV Accuracy (Training)": [cv_mean_accuracy],
        "CV Std (Training)": [cv_std],
        "Accuracy (Test)": [test_accuracy],
        "F1 Score (Test)": [test_f1_score],
        "Precision Score (Test)": [test_precision_score],
    })

    return results
