"""
Author: Reto Hendry
Date: 2023-04-08

This is gathered code from histology_analysis_on_MVPA.ipynb
"""

import numpy as np
import pandas as pd
import tqdm

def get_best_features_sorted(array, df_label, feature_list_values_=False, proof_of_correctness_arg=False):
    """
    Does a (cumulative) histogram analysis to find the features/voxels 
    which differ the most across the two classes (treatment / no treatment).
    Returns an array with all the features sorted according to its relevance.

    Parameters
    ----------
    array : 4D-array [samples, x, y, z]
        The resampled array with the MVPA data.
    df_label : df
        The dataframe with Ground Truth, column name "Cond".
        value 1 = "treatment received", 
        value 0 = "treatment not received"
    feature_list_values_ : bool, optional
        If True, the function returns a list with the 
        values of the features, by default False. 
    proof_of_correctness_arg : bool, optional
        This parameter is only set to true to test the entire pipeline,
        by default False.

    Return
    ------
    feature_list : list
        List with all feature columns sorted according their relevance. 
    feature_list_values : list, optional
        only returned when feature_list_values_ = True. 
        List with all feature values sorted according their relevance.
        They match with the feature list. 
    """

    if proof_of_correctness_arg:
        # change the values of the array to zeros
        array = np.zeros(array.shape)

        # For the samples which received the treatment (df_label["Cond"] = 1):
        # change the middle plane of every sample to 1
        array[df_label["Cond"]==1, int(array.shape[1]/2), :, :] = 1
        # set a volume of (5, 5, 5) of the lower corner of every sample to 1
        array[df_label["Cond"]==1, 0:5, 0:5, 0:5] = 1

    # flatten the array into a 2d array (keep the sample dimension)
    array_2d = array.reshape(array.shape[0], -1)
        
    # create a dataframe from the 2d array
    df_small = (pd.DataFrame(array_2d))

    # min and max values of the entire array to set the same range for all histograms (to make sure the binning is the same)
    max_value = np.max(df_small.values)
    min_value = np.min(df_small.values)
    
    # calculate the difference between the two cumulative histograms including 
    # a resampling (done with the range argument) so that the two histograms have the same bins
    print("Calculation for list of best features ...")
    cum_diff_dict = {}
    for col in tqdm.tqdm(df_small.columns[:-1]):  # -1 because the last column is the ground truth
        cum_diff = np.abs(np.cumsum(np.histogram(df_small.loc[df_label["Cond"]==1 , col],
                                                    bins=100, density=True,
                                                    range=(min_value, max_value)  # resample to have same binning
                                                    )[0])
                        - np.cumsum(np.histogram(df_small.loc[df_label["Cond"]==0 , col],
                                                    bins=100, density=True,
                                                    range=(min_value, max_value)  # resample to have same binning
                                                    )[0]))
        cum_diff_dict[col] = cum_diff.sum()
    
    # sort the list of tuples according to the most relevant feature (biggest cumsum)
    cum_diff_list = sorted(cum_diff_dict.items(), key=lambda x:x[1], reverse=True)  # creates a list of tuples

    # create a list with the index and values of the features
    feature_list = [x[0] for x in cum_diff_list]
    feature_list_values = [x[1] for x in cum_diff_list]


    if feature_list_values_:
        return feature_list, feature_list_values
    else:
        return feature_list