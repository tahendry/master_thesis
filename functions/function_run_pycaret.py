import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score
from pycaret.classification import setup, compare_models, tune_model, finalize_model, predict_model


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
        data_split_stratify = True,  # stratifistratifiedkfolded split
        normalize = True,  # normalize data
        normalize_method = "zscore",  #  z = (x - u) / s
        fold_strategy = "stratifiedkfold",
        fold = 5,  # number of folds for CV
        fold_shuffle = True,  # shuffle of CV
        n_jobs = -1,  # use all processors
        use_gpu = True,  # use GPU for some models if available
        session_id = 1,  # random state
        verbose = True,  # print information grid
    )

    # Compare and select the best model
    best_model = compare_models()
    metrics_best_model = pull()

    # Tune the best model
    tuned_model = tune_model(best_model)
    metrics_tuned_model = pull()

    # Finalize the model
    final_model = finalize_model(tuned_model)

    # Perform cross-validation on the training set
    cv_scores = cross_val_score(final_model, 
                                X=train_data.drop("Cond", axis=1), 
                                y=train_data["Cond"],
                                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                n_jobs=-1, 
                                error_score='raise')
    cv_mean_accuracy = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    # Make predictions on the test set
    predictions = predict_model(final_model, data=test_data)

    # Calculate the accuracy and F1 score on the test set
    test_accuracy = accuracy_score(test_data["Cond"], predictions["Cond"])
    test_f1_score = f1_score(test_data["Cond"], predictions["Cond"], average="weighted")
    test_precision_score = precision_score(test_data["Cond"], predictions["Cond"], average="weighted")

    # Create a DataFrame to store the results
    id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
