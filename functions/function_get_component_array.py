"""
Author: Reto Hendry
Date: 2023-04-08
"""

import os
import numpy as np
import nibabel as nib


def get_component_array(components, print_info=False, data_path_optional=None):
    """
    Returns an arrays with the stacked components of the MVPA data.
    (component, samples, x, y, z)

    Parameters
    ----------
    components : list
        list of components to be used
    print_info : bool, optional
        print info about the array, by default False
    data_path_optional : string, optional
        Path to the data folder. The default is None.
        This is only used when the function is called from
        a script inside a folder other than the main folder.

    Returns
    -------
    inpt_diff_stacked : array
        array with the stacked components of the MVPA data
        (components, samples, x, y, z)

    """

    # read MVPA data
    if data_path_optional:
        data_path = data_path_optional
    else:
        data_path = "../data/"
    path_content = os.listdir(os.path.join(data_path, "Denoised_MVPA_8mm"))

    components = sorted(components)
    comp_diff_list = []

    for component in components:
        
        sample_diff_list = []
        # make two lists with pre (Condition001) and post (Condition002) data 
        # sort them to make sure they are in order
        # example:  'BETA_Subject001_Condition001_Measure001_Component001.nii', 
        #           'BETA_Subject002_Condition001_Measure001_Component001.nii',
        #           'BETA_Subject003_Condition001_Measure001_Component001.nii', ...
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