"""
Date: 09.04.2023
Author: Reto Hendry

This scripts allows to run a parameter sweep on the MVPA data using the autoML tool pycaret.
The results are stored in results_param_sweep_pycaret.csv.

The following parameters need to be defined:
- list_of_components (list of integers)
    the integer represents the component of the MVPA data which should be used
- resample_cube_list (list of integers)
    the integer represents the size of the cube which is used to resample the MVPA data
- number_of_feature_list (list of integers)
    the integer represents the number of (best) features which should be used for the classification
- randomize_label (boolean)
    if True, the labels are mixed randomly before the feature selection
    This is to check how the results change when the labels are mixed randomly.

"""

import numpy as np
import pandas as pd

from functions.function_get_label_df import get_label_df
from functions.function_get_component_array import get_component_array
from functions.function_get_best_features import get_best_features_sorted
from functions.function_resample_4d_array import resample_4d_array
from functions.function_run_pycaret import run_pycaret


##############################################

# parameters to define
list_of_components = [1]
resample_cube_list = np.arrange(1, 6, 1)
number_of_feature_list = np.arrange(10, 411, 20)
randomize_label = False

##############################################


# get the label data
df_label = get_label_df()

# get the MVPA data arrays
component_array_5d = get_component_array(list_of_components)
print(f"shape of component_array_5d: {component_array_5d.shape}")

# create empty dataframe to store results
try:
    result_df = pd.read_csv("./results/results_param_sweep_pycaret.csv")
except:
    result_df = pd.DataFrame()

# loop through different cube sizes
for resample_cube in resample_cube_list:

    # loop through the components
    for indx, component in enumerate(list_of_components):

        sample_array_4d = resample_4d_array(component_array_5d[indx], resample_cube)
        print(f"shape of resampled sample_array_4d: {sample_array_4d.shape}")

        # get the sorted feature list
        best_feature_list = get_best_features_sorted(
            sample_array_4d, df_label
        )

        # loop through the number of features
        for number_of_features in number_of_feature_list:
            
            # select desired number of features
            best_features = best_feature_list[:number_of_features]

            # run pycaret on the best features
            single_result_df = run_pycaret(
                sample_array_4d, df_label, best_features, component, resample_cube, randomize_labels=randomize_label
            )
            result_df = pd.concat([result_df, single_result_df], axis=0)

            # save the result_df to csv
            result_df.to_csv("./results/results_param_sweep_pycaret.csv", index=False)
