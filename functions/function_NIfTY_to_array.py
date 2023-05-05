"""
Date: 05.05.2023
Author: Reto Hendry

"""

import nibabel as nib
import os

def nifty_to_array(file_path_from_data):
    """
    This function loads a nifty file and returns the data as a numpy array.

    Parameters
    ----------
    file_path_from_data : str
        The file path to the nifty file relative to the folder "data".

    Return
    ------
    array : 3D-array [x, y, z]
        The array with the data.

    """

    file_path = "../data/"

    brain_vol = nib.load(os.path.join(file_path, file_path_from_data))

    brain_vol_data = brain_vol.get_fdata()

    return brain_vol_data


