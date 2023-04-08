"""
Author: Reto Hendry
Date: 2023-04-08

This is gathered code from histology_analysis_on_MVPA.ipynb
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from function_get_label_df import get_label_df

def get_best_features(array, df_label, number_of_features):
    """
    Returns an array with the best features of the MVPA data.

    Parameters
    ----------
    array : 4D-array [samples, x, y, z]
        The array with the MVPA data.
    number_of_features : int
        The number of features to be returned.
    """

    # print the array shape before flattening
    print(f"shape of array before flattening: {array.shape}")

    # flatten the array into a 2d array (keep the sample dimension)
    array_2d = array.reshape(array.shape[0], -1)
    print(f"shape of array_2d: {array_2d.shape}")

    # create a dataframe from the 2d array
    df_small = (pd.DataFrame(array_2d))

    # min and max values of the entire array to set the same range for all histograms (to make sure the binning is the same)
    max_value = np.max(df_small.values)
    min_value = np.min(df_small.values)
    
    # calculate the difference between the two cumulative histograms including 
    # a resampling (done with the range argument) so that the two histograms have the same bins
    cum_diff_dict = {}
    for col in df_small.columns[:-1]:  # -1 because the last column is the ground truth
        cum_diff = np.abs(np.cumsum(np.histogram(df_small.loc[df_label["Cond"]==1 , col],
                                                    bins=100, density=True,
                                                    range=(min_value, max_value)  # resample to have same binning
                                                    )[0])
                        - np.cumsum(np.histogram(df_small.loc[df_label["Cond"]==0 , col],
                                                    bins=100, density=True,
                                                    range=(min_value, max_value)  # resample to have same binning
                                                    )[0]))
        cum_diff_dict[col] = cum_diff.sum()
        
    cum_diff_list = sorted(cum_diff_dict.items(), key=lambda x:x[1], reverse=True)  # creates a list of tuples

    top_percent = number_of_features / len(array_2d[1])

    # create a list with the top_percent biggest differences
    feature_list = [x[0] for x in cum_diff_list[:int(len(cum_diff_list)*top_percent)]]
    print(f"{len(feature_list)} features selected")