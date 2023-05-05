"""
Date: 09.04.2023
Author: Reto Hendry

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
from functions.function_plot_3d_array_colored import plot_3d_array_colored
from functions.function_get_feature_positions import get_feature_positions
from functions.function_NIfTY_to_array import nifty_to_array

##############################################

# parameters to define
proof_of_correctness = True
list_of_components = [1]
resample_cube_list = [5]
number_of_feature_list = [2]

if resample_cube_list[0] == 5:
    padding = [(2, 2), (0, 1), (2, 2)]
    shape_resampeled_array = (19, 22, 19)

##############################################


# get the label data
df_label = get_label_df()

# get the MVPA data arrays
component_array_5d = get_component_array(list_of_components)
print(f"shape of component_array_5d: {component_array_5d.shape}")

# create empty dataframe to store results
try:
    result_df = pd.read_csv("./results/results_proof.csv")
except:
    result_df = pd.DataFrame()

# loop through different cube sizes
for reshape_cube in resample_cube_list:

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

        sample_array_4d = resample_4d_array(component_array_5d[indx], reshape_cube)
        print(f"shape of resampled sample_array_4d: {sample_array_4d.shape}")

        if proof_of_correctness:
            # change the values of the array to zeros
            array = np.zeros(sample_array_4d.shape)

            # For the samples which received the treatment (df_label["Cond"] = 1):
            # change the middle plane of every sample to 1
            # array[df_label["Cond"]==1, int(array.shape[1]/2), :, :] = 1
            # set a volume of (5, 5, 5) of the lower corner of every sample to 1
            array[df_label["Cond"]==1, 0, 0, 0] = 1
            sample_array_4d = array

        # get the sorted feature list
        best_feature_list = get_best_features_sorted(
            sample_array_4d, df_label
        )

        # loop through the number of features
        for number_of_features in number_of_feature_list:
            
            # select desired number of features
            best_features = best_feature_list[:number_of_features]

            # print the positions of the features on to the MVPA data
            marker = get_feature_positions(best_features, padding, resample_cube_list[0], shape_resampeled_array)
            plot_3d_array_colored(component_array_5d[0][0], marker)

            # print the positions of the features onto the anatomical data
            anatomic_array = nifty_to_array("anatomic_scan_T1.nii")
            plot_3d_array_colored(anatomic_array, marker)

            # run h2o on the best features
            single_result_df = run_h2o(
                sample_array_4d, df_label, best_features, component, reshape_cube
            )
            result_df = pd.concat([result_df, single_result_df], axis=0)

            # save the result_df to csv
            result_df.to_csv("./results/results_proof.csv", index=False)

    # stop h2o server
    h2o.shutdown(prompt=False)
    time.sleep(60)

# show the result_df
result_df