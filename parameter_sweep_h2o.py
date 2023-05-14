"""
Date: 09.04.2023
Author: Reto Hendry

This scripts allows to run a parameter sweep on the MVPA data using the autoML tool h2o.
The results are stored in results_param_sweep_h2o.csv.

The following parameters need to be defined:
- list_of_components (list of integers)
    the integer represents the component of the MVPA data which should be used
- resample_cube_list (list of integers)
    the integer represents the size of the cube which is used to resample the MVPA data
- number_of_feature_list (list of integers)
    the integer represents the number of (best) features which should be used for the classification

"""

import numpy as np
import pandas as pd
import h2o
import time

from functions.function_get_label_df import get_label_df
from functions.function_get_component_array import get_component_array
from functions.function_get_best_features import get_best_features_sorted
from functions.function_resample_4d_array import resample_4d_array
from functions.function_run_h2o import run_h2o


##############################################

# parameters to define
list_of_components = [1]
resample_cube_list = np.arange(1, 16, 1, dtype=int)
number_of_feature_list = np.arange(10, 411, 10, dtype=int)

##############################################


# get the label data
df_label = get_label_df()

# get the MVPA data arrays
component_array_5d = get_component_array(list_of_components)
print(f"shape of component_array_5d: {component_array_5d.shape}")

# create empty dataframe to store results
try:
    result_df = pd.read_csv("./results/results_param_sweep_h2o.csv")
except:
    result_df = pd.DataFrame()

# loop through different cube sizes
for resample_cube in resample_cube_list:

    # start h2o server
    # It was noticed that if the server is not restarted from time to time,
    # the training of the models becomes slower and slower.
    h2o.init(
        ip="localhost", 
        port=54323,
        nthreads=-1,
        min_mem_size=64,  # 64 GB
        max_mem_size=160,  # 160 GB
    )

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

            # run h2o on the best features
            single_result_df = run_h2o(
                sample_array_4d, df_label, best_features, component, resample_cube
            )
            result_df = pd.concat([result_df, single_result_df], axis=0)

            # save the result_df to csv
            result_df.to_csv("./results/results_param_sweep_h2o.csv", index=False)

    # stop h2o server
    h2o.shutdown(prompt=False)
    time.sleep(60)
