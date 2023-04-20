"""
Date: 09.04.2023
Author: Reto Hendry

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
reshape_cube_list = [5]
number_of_feature_list = np.arange(10, 200, 25, dtype=int)

##############################################


# get the label data
df_label = get_label_df()

# get the MVPA data arrays
component_array_5d = get_component_array(list_of_components)
print(f"shape of component_array_5d: {component_array_5d.shape}")

# create empty dataframe to store results
try:
    result_df = pd.read_csv("./results/pycaret_results_df.csv")
except:
    result_df = pd.DataFrame()

# loop through different cube sizes
for reshape_cube in reshape_cube_list:

    # loop through the components
    for indx, component in enumerate(list_of_components):

        sample_array_4d = resample_4d_array(component_array_5d[indx], reshape_cube)
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
                sample_array_4d, df_label, best_features, component, reshape_cube
            )
            result_df = pd.concat([result_df, single_result_df], axis=0)

            # save the result_df to csv
            result_df.to_csv("./results/pycaret_results_df.csv", index=False)

# show the result_df
result_df