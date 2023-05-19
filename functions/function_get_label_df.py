"""
Author: Reto Hendry
Date: 2023-04-08
"""


import os
import pandas as pd


def get_label_df(data_path_optional=None):
    """
    Read in the excel-file with the labels and 
    return a dataframe with the labels.

    Parameters
    ----------
    data_path_optional : string, optional
        Path to the data folder. The default is None.
        This is only used when the function is called from
        a script inside a folder other than the main folder.

    Returns
    -------
    df_label : pandas dataframe
        Dataframe with the labels.
        Column with condition/label: "Cond"
        Sham (placebo) = 0, Verum = 1

    """

    # read in the excel-file with the labels
    if data_path_optional:
        data_path = data_path_optional
    else:
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