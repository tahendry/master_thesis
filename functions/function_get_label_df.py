"""
Author: Reto Hendry
Date: 2023-04-08
"""


import os
import pandas as pd


def get_label_df():
    """
    Read in the excel-file with the labels and 
    return a dataframe with the labels.

    Parameters
    ----------
    None

    Returns
    -------
    df_label : pandas dataframe
        Dataframe with the labels.
        Column with condition/label: "Cond"
        Sham (placebo) = 0, Verum = 1

    """
    # read in the excel-file with the labels
    data_path = "../data/"
    label_file = "Conn_IDs_Matching_90_subjects.xlsx"

    # read excel with only the first three columns
    # Sham = 1, Verum = 2
    df_label = (pd.read_excel(os.path.join(data_path, label_file),
                                usecols=[0, 1, 2])
                .replace({"Cond": {1: 0}})
                .replace({"Cond": {2: 1}})
                )

    df_label["Cond"] = df_label["Cond"].astype("category")

    return df_label