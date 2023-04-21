"""
Date: 09.04.2023
Author: Reto Hendry

"""


import numpy as np
import pandas as pd
import datetime
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score


def run_tpot(sample_array_4d, df_label, feature_list, component, reshape_cube):
    """
    Run tpot on the input data.

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
        a dataframe with the results of the best model of the tpot run

    """
    
    # Define the cross-validation strategy
    cv_stratified = StratifiedKFold(n_splits=9, shuffle=True, random_state=42)

    # reshape 4d array to dataframe
    sample_df = pd.DataFrame(
        sample_array_4d.reshape(sample_array_4d.shape[0], -1)
        ).iloc[:, feature_list]

    # split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(
        sample_df, 
        df_label["Cond"], 
        test_size=0.2, 
        random_state=42,
        stratify=df_label["Cond"]
    )

    def define_tpot_model(
        nbr_generations=100, nbr_population_size=100, max_time=120
    ):

        tpot_model = TPOTClassifier(
            generations=nbr_generations,  # number of iterations for optimization
            population_size=nbr_population_size,  # default 100
            max_time_mins=max_time,
            early_stop=10,  # number of iterations without improvement before stopping
            scoring="accuracy",
            cv=cv_stratified,  # cross validation fold (default)
            n_jobs=-1, # nbr. of cores used (-1 = all)
            max_eval_time_mins=10,  # default 5
            random_state=1,  # seed
            memory=False,  # avoid fitting same model
            verbosity=2,  # 0 = minimal info, 3 = max info
            # periodic_checkpoint_folder=f"./tpot_folder_reduced_topF",
            # log_file="./tpot_folder/tpot_log",
        )
        return tpot_model
    
    tpot = define_tpot_model()

    # Fit the TPOT classifier on the training set
    print(f"Fitting tpot model on {len(feature_list)} features on component {component}...")
    tpot.fit(x_train, y_train)

    # Access the best model (pipeline)
    best_model = tpot.fitted_pipeline_

    # Save best model
    id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tpot.export(f"./tpot_best_models/tpot_pipeline_{id}.py")

    # Perform cross-validation on the training set
    cv_scores = cross_val_score(best_model, x_train, y_train, cv=cv_stratified)
    cv_mean_accuracy = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    # Make predictions on the test set
    y_pred = best_model.predict(x_test)

    # Calculate the accuracy and F1 score on the test set
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1_score = f1_score(y_test, y_pred, average="weighted")
    test_precision_score = precision_score(y_test, y_pred, average="weighted")

    # Create a DataFrame to store the results
    results = pd.DataFrame({
        "id": [f"tpot_{id}"],
        "component": [component],
        "resample_cube": [reshape_cube],
        "number_of_features": [len(feature_list)],
        "CV_Accuracy_(Training)": [cv_mean_accuracy],
        "CV_Std_(Training)": [cv_std],
        "Accuracy_(Test)": [test_accuracy],
        "F1_Score_(Test)": [test_f1_score],
        "Precision_Score_(Test)": [test_precision_score],
    })
    
    return results