"""
Author: Reto Hendry
Date: 2023-04-08
"""

import os
import numpy as np
import nibabel as nib


def get_component_array(components, print_info=False):
    """
    Returns an arrays with the stacked components of the MVPA data.

    Parameters
    ----------
    component : list of components
    """

    # read MVPA data
    data_path = "../data/"
    path_content = os.listdir(os.path.join(data_path, "Denoised_MVPA_8mm"))

    components = sorted(components)
    comp_diff_list = []

    for component in components:
        sample_diff_list = []
        # make two lists with pre (Condition002) and post (Condition003) data 
        # of component in loop
        pre = sorted([x for x in path_content 
                            if f"Component00{component}" in x 
                            and "Condition001" in x])
        post = sorted([x for x in path_content 
                            if f"Component00{component}" in x 
                            and "Condition002" in x])
        
        if print_info:
            print(f"there are {len(pre)} pre and {len(post)} post samples for component {component}")

        # loop over pre and post samples and calculate difference
        for pre, post in zip(pre, post):
            pre_vol = nib.load(
                os.path.join(data_path, "Denoised_MVPA_8mm", pre)
            )
            post_vol = nib.load(
                os.path.join(data_path, "Denoised_MVPA_8mm", post)
            )
            pre_vol_data = pre_vol.get_fdata()
            post_vol_data = post_vol.get_fdata()
            diff_vol_data = post_vol_data - pre_vol_data  # type = array
            sample_diff_list.append(diff_vol_data)

        comp_diff_list.append(sample_diff_list)

    # this stacks the two lists on top of each other
    # resulting in [component, sample, x, y, z]
    inpt_diff_stacked = np.stack(comp_diff_list, axis=0)

    if print_info:
        print(f"type of single volume array {type(diff_vol_data)}")
        print(f"shape of single volume array {diff_vol_data.shape}")
        print(f"shape of stacked array {inpt_diff_stacked.shape}")

    return inpt_diff_stacked