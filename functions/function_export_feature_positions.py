"""
Date: 05.05.2023
Author: Reto Hendry

"""

import numpy as np
import nibabel as nib
import os

def export_feature_positions(array_shape, marker, output_path, output_filename, feature_importance=False):
    """
    Create a NIfTI file in the same format as the anatomic_scan_T1.nii file.
    If feature_importance is False:
        The voxels which are marked via the marker argument have the maximum 
        value (white color) for the data type which was used in the original NIfTI file.
        The voxels which are not marked, have a value of 0 (black color)
    If feature_importance is True:
        The voxels which are marked via the marker argument have the feature importance value
        multiplied by 255 (white color). That means a scaled feature importance of 1 will be
        white and a scaled feature importance of 0 will be black.

    Parameters
    ----------
    array_shape : tuple
        The shape of the array.
    marker : list
        List with the indices of the voxels which should be marked.
    output_filename : str
        The filename of the output file.
    output_path : str
        The path where the output file will be saved.
        (relative to the file which is calling this function)
    feature_importance : list
        List with the scaled feature importance values for each feature/bin.

    Return
    ------
    None -> save the NIfTI file

    """

    # Load the original NIfTI file (only the header and affine matrix)
    if feature_importance:
        file_path = "../example_data/"
    else:
        file_path = "./example_data/"
    original_img = nib.load(os.path.join(file_path, "anatomical_scan_T1.nii"))
    original_header = original_img.header.copy()
    original_affine = original_img.affine

    # Replace this with your 3D numpy array
    black_image = np.zeros((91, 109, 91), dtype=np.uint8)

    # check if shape maches
    assert black_image.shape == array_shape, "shape does not match!"

    if feature_importance:
        # Set the indices in the marker array to the feature importance value
        feature_importance = [item for item in feature_importance for _ in range(3*3*3)]
        for k, idx in enumerate(marker):
            black_image[tuple(idx)] = int(feature_importance[k] * 255)
    else:
        # Set the indices in the marker array to white (255 for uint8)
        for idx in marker:
            black_image[tuple(idx)] = 255

    # this would be the affine matrix of the original image
    # affine_matrix = np.array([
    #     [-2,  0,  0, 90],
    #     [ 0,  2,  0, -126],
    #     [ 0,  0,  2, -72],
    #     [ 0,  0,  0, 1]
    # ])

    # Create a new NIfTI image with the same affine matrix as the original
    nifti_img = nib.Nifti1Image(black_image, original_affine, header=original_header)

    # Modify only the necessary attribute in the header
    header = nifti_img.header
    nifti_img.header.set_data_dtype(np.uint8)
    header['scl_slope'] = 0
    header['scl_inter'] = 0
    header['descrip'] = b'Mask of most important voxels for classification'

    # Save the modified NIfTI image to a file
    nib.save(nifti_img, os.path.join(output_path, f"{output_filename}.nii"))

    print(f"saved {output_filename}.nii successfully")
