"""
Date: 09.04.2023
Author: Reto Hendry

This script is used to proof the correctness of the feature selection algorithm.

The data is manipulated in a way that the feature selection algorithm should select 
the middle plane and the corner at index (0, 0, 0) of the samples as the best feature.
This is done by setting the middle plane and the corner of the samples which received 
the treatment to 1 and the rest to 0. Therefore they are easily distinguishable and 
should be chose as the best features. As a result, the accuracy of the model should
be 100%, which is the case.

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
from functions.function_export_feature_positions import export_feature_positions

##############################################

# parameters to define
list_of_components = [1]
resample_cube_list = [1]
plots_3d = True
run_autoML = False

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
for resample_cube in resample_cube_list:

    if run_autoML:
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

        sample_array_4d, padding = resample_4d_array(component_array_5d[indx], resample_cube, return_padded_array=True)
        print(f"shape of resampled sample_array_4d: {sample_array_4d.shape}")

        # change the values of the array to zeros
        array = np.zeros(sample_array_4d.shape)

        # For the samples which received the treatment (df_label["Cond"] = 1):
        # change the middle plane of every sample to 1
        array[df_label["Cond"]==1, int(array.shape[1]/2), :, :] = 1
        # set a volume of (5, 5, 5) of the lower corner of every sample to 1
        array[df_label["Cond"]==1, 0, 0, 0] = 1
        sample_array_4d_for_proof = array

        # get the sorted feature list
        best_feature_list = get_best_features_sorted(
            sample_array_4d_for_proof, df_label
        )

        number_of_feature_list = [sample_array_4d[0].shape[0] * sample_array_4d[0].shape[1] + 1]

        # loop through the number of features
        for number_of_features in number_of_feature_list:
            
            # select desired number of features
            best_features = best_feature_list[:number_of_features]

            # print the positions of the features on to the MVPA data
            marker, marker_array_shape = get_feature_positions(best_features, padding, resample_cube_list[0], sample_array_4d_for_proof[0].shape)
            if plots_3d:
                plot_3d_array_colored(component_array_5d[0][0], plot_name=f"./figures/proof_of_corr_rc{resample_cube}")
                plot_3d_array_colored(component_array_5d[0][0], marker, plot_name=f"./figures/proof_of_corr_rc{resample_cube}_marker")

            # print the positions of the features onto the anatomical data
            anatomical_array = nifty_to_array("anatomical_scan_T1.nii")
            if plots_3d:
                plot_3d_array_colored(anatomical_array, marker, plot_name=f"./figures/proof_of_corr_rc{resample_cube}_anatomical_marker")

            if run_autoML:
                # run h2o on the best features
                single_result_df = run_h2o(
                    sample_array_4d_for_proof, df_label, best_features, component, resample_cube
                )
                result_df = pd.concat([result_df, single_result_df], axis=0)

                # save the result_df to csv
                result_df.to_csv("./results/results_proof.csv", index=False)

            # export the feature positions to a NIfTI file
            export_feature_positions(array_shape=marker_array_shape, 
                                     marker=marker,
                                     output_path="./NIfTY_feature_masks",
                                     output_filename=f"proof_mask_c{component}_rc{resample_cube}_nof{number_of_features}")

    if run_autoML:
        # stop h2o server
        h2o.shutdown(prompt=False)
