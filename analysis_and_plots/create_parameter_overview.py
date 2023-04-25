"""
Date: 09.04.2023
Author: Reto Hendry

"""

import numpy as np
import pandas as pd
import sys

sys.path.append("/home/tahendry/Documents/master_thesis/master_thesis_repo")
from functions.function_get_label_df import get_label_df
from functions.function_get_component_array import get_component_array
from functions.function_get_best_features import get_best_features_sorted
from functions.function_resample_4d_array import resample_4d_array


##############################################

# parameters to define
list_of_components = [1]
resample_cube_list = np.arange(1, 16, 1)
number_of_feature_list = np.arange(10, 211, 10, dtype=int)

##############################################


### create a dataframe with the best features for each parameter combination
feature_list_df = pd.DataFrame()

# get the label data
df_label = get_label_df()

# load feature_list_df if it exists
try:
    feature_list_df = pd.read_csv("./results/parameter_overview_df.csv")
except:
    # get the MVPA data arrays
    component_array_5d = get_component_array(list_of_components)
    print(f"shape of component_array_5d: {component_array_5d.shape}")

    # loop through different cube sizes
    for reshape_cube in resample_cube_list:

        # loop through the components
        for indx, component in enumerate(list_of_components):

            sample_array_4d, padding = resample_4d_array(component_array_5d[indx], reshape_cube, return_padded_array=True)
            print(f"shape of resampled sample_array_4d: {sample_array_4d.shape}")

            # get the sorted feature list
            best_feature_list = get_best_features_sorted(
                sample_array_4d, df_label
            )

            # loop through the number of features
            for number_of_features in number_of_feature_list:
                
                # select desired number of features
                best_features = best_feature_list[:number_of_features]

                # add the best features to the dataframe
                df_temp = pd.DataFrame(
                    {
                        "component": [component],
                        "reshape_cube": [reshape_cube],
                        "padding": [padding],
                        "number_of_features": [number_of_features],
                        "shape_resampled_array": [sample_array_4d.shape],
                        "best_features": [best_features],
                    },
                )
                df_temp = pd.DataFrame(df_temp, index=[0])

                feature_list_df = pd.concat([feature_list_df, df_temp], axis=0, ignore_index=True)

    # save the feature_list_df
    feature_list_df.to_csv("./results/parameter_overview_df.csv", index=False)

feature_list_df



