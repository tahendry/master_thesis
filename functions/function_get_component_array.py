"""
Author: Reto Hendry
Date: 2023-04-08
"""

import os
import numpy as np
import nibabel as nib


def get_component_array(components):
    """
    Returns an arrays with the stacked components of the MVPA data.

    Parameters
    ----------
    component : list of components
    """

    # read MVPA data
    data_path = "../data/"
    path_content = os.listdir(os.path.join(data_path, "Denoised_Data_6mm", "MVPA_data"))

    components = sorted(components)
    inpt_diff_list = []

    for component in components:
        diff = []
        # make two lists with pre (Condition002) and post (Condition003) data of first component
        pre = sorted([x for x in path_content 
                            if f"Component00{component}" in x 
                            and "Condition002" in x])
        post = sorted([x for x in path_content 
                            if f"Component00{component}" in x 
                            and "Condition003" in x])

        for pre, post in zip(pre, post):
            pre_vol = nib.load(os.path.join(data_path, "Denoised_Data_6mm", "MVPA_data", pre))
            post_vol = nib.load(os.path.join(data_path, "Denoised_Data_6mm", "MVPA_data", post))
            pre_vol_data = pre_vol.get_fdata()
            post_vol_data = post_vol.get_fdata()
            diff_vol_data = post_vol_data - pre_vol_data
            diff.append(diff_vol_data)

        inpt_diff_list.append(diff)

    # check the type of the data
    print(f"{type(diff[0])=}")

    # stack the data to later use it as input for the CNN
    # note: the first dimension is the number of samples
    print(f"shape of one list element before stacking: {diff[0].shape=}")
    inpt_diff = np.stack(diff, axis=0)

    print(f"{inpt_diff.shape=}")

    return np.stack(inpt_diff_list, axis=0)