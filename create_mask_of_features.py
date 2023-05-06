"""
Date: 09.04.2023
Author: Reto Hendry

This script creates NIfTI files which show the positions of the most important features.
It can be used to show where they are by overlaying them onto the anatomical scan.
Additionally, they can be used as a mask in other tools to do further analysis.

To run this scripts, the parameter_overview_df.csv file is needed. It can be created
by running the parameter_overview.py script.

"""

import pandas as pd
import ast
from functions.function_get_feature_positions import get_feature_positions
from functions.function_export_feature_positions import export_feature_positions


##############################################

# read in the parameter overview dataframe
param_overview_df = pd.read_csv("./results/parameter_overview_df.csv")

# Convert string representations to their corresponding data structures
param_overview_df['padding'] = param_overview_df['padding'].apply(ast.literal_eval)
param_overview_df['shape_resampled_array'] = param_overview_df['shape_resampled_array'].apply(ast.literal_eval)
param_overview_df['best_features'] = param_overview_df['best_features'].apply(ast.literal_eval)

for index, row in param_overview_df.iterrows():

    # create the marker array
    marker, marker_array_shape = get_feature_positions(
        top_features=row["best_features"],
        padding=row["padding"],
        resample_cube=row["resample_cube"],
        shape_resampled_array=row["shape_resampled_array"][-3:],
    )

    # export the feature positions to a NIfTI file
    component = row["component"]
    reshape_cube = row["resample_cube"]
    number_of_features = row["number_of_features"]
    output_path = "./NIfTY_feature_masks"
    export_feature_positions(
        array_shape=marker_array_shape, 
        marker=marker, 
        output_path=output_path,
        output_filename=f"mask_c{component}_rc{reshape_cube}_nof{number_of_features}"
    )
