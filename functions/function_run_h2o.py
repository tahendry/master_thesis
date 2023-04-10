"""
Date: 09.04.2023
Author: Reto Hendry

"""

import numpy as np
import pandas as pd
import datetime
import h2o
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.base import BaseEstimator


def run_h2o(sample_array_4d, df_label, feature_list, component, reshape_cube):
    """
    Run H2O AutoML on the input data.

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
        a dataframe with the results of the best model of the H2O AutoML run

    """

    def cross_val_metrics(model, X, y, cv=5):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        accuracy_scores = []
        
        for train_index, val_index in skf.split(X, y):
            X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
            y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

            X_train_cv_h2o = h2o.H2OFrame(pd.concat([X_train_cv, y_train_cv], axis=1))
            X_val_cv_h2o = h2o.H2OFrame(pd.concat([X_val_cv, y_val_cv], axis=1))

            X_train_cv_h2o[y_label] = X_train_cv_h2o[y_label].asfactor()
            X_val_cv_h2o[y_label] = X_val_cv_h2o[y_label].asfactor()

            val_pred = model.predict(X_val_cv_h2o)
            y_val_pred = val_pred.as_data_frame().values.argmax(axis=1)
            
            accuracy = accuracy_score(y_val_cv, y_val_pred)
            accuracy_scores.append(accuracy)
        
        mean_accuracy = np.mean(accuracy_scores)
        std_accuracy = np.std(accuracy_scores)
        
        return mean_accuracy, std_accuracy
    
    h2o.init()

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
    
    # Convert the column names of x_train and x_test to strings
        # the conversion into strings is necessary for the H2O AutoML
        # the h2o dataframes convert all types to strings
    x_train.columns = x_train.columns.astype(str)
    x_test.columns = x_test.columns.astype(str)

    # Convert train and test sets to H2O DataFrames 
    x_train_h2o = h2o.H2OFrame(pd.concat([x_train, y_train], axis=1))
    x_test_h2o = h2o.H2OFrame(pd.concat([x_test, y_test], axis=1))

    x_features = x_train.columns.tolist()
    y_label = "Cond"

    x_train_h2o[y_label] = x_train_h2o[y_label].asfactor()
    x_test_h2o[y_label] = x_test_h2o[y_label].asfactor()

    # define the autoML model
    h2o_model = H2OAutoML(
        max_runtime_secs=7200, 
        max_models=100, 
        seed=1,
        balance_classes=True,
        stopping_rounds=10,  # stop training if the score doesn't improve for 10 rounds
        verbosity="info",
    )

    h2o_model.train(x=x_features, y=y_label, training_frame=x_train_h2o)

    best_model = h2o_model.leader

    # save best model
    id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    h2o.save_model(
        model=best_model, 
        path=f"./h2o_best_models/h2o_model_{id}",
        force=True,  # overwrite existing model
    )

    ### Calculate cross-validated metrics from training data
    # cv-accuracy and cv-standard deviation from best model
    cv_mean_accuracy, cv_std = cross_val_metrics(best_model, x_train, y_train)

    ### calculate test metrics
    # Make predictions on the test set
    test_pred = best_model.predict(x_test_h2o)
    y_test_pred = test_pred.as_data_frame().values.argmax(axis=1)

    # Calculate the accuracy and F1 score on the test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1_score = f1_score(y_test, y_test_pred, average="weighted")
    test_precision_score = precision_score(y_test, y_test_pred, average="weighted")

    # Create a DataFrame to store the results
    results = pd.DataFrame({
        "id": [f"h2o_{id}"],
        "component": [component],
        "reshape_cube": [reshape_cube],
        "number of features": [len(feature_list)],
        "CV Accuracy (Training)": [cv_mean_accuracy],
        "CV Std (Training)": [cv_std],
        "Accuracy (Test)": [test_accuracy],
        "F1 Score (Test)": [test_f1_score],
        "Precision Score (Test)": [test_precision_score],
    })

    h2o.shutdown(prompt=False)
    
    return results