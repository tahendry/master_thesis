"""
Author: Reto Hendry
Date: 2023-03-25
"""

import numpy as np
import tqdm

def resample_4d_array(big_array_4d, reshape_cube, print_array_sizes=False, return_padded_array=False):
    """
    Resamples each 3d array for each sample (4th dimension) in a 4d array.
    The resampling is done by taking the mean of non-overlapping volumes of
    a specified size. The input array is first padded to ensure that
    it can be evenly divided into the smaller volumes.
    
    This function contains another function.
    The function "resample_3d_array" inside this function is defined first.
    It is used to resample a 3D array (x, y, z), or in other words a single sample.
    
    Parameters:
    ----------
    big_array_4d : numpy array
        The bigger array which is to be reshaped.
    reshape_cube : int
        The size of the volume (SxSxS) which 
        is to be used to calculate the mean
        of the smaller array.
    print_array_sizes : bool
        If True, the size of the original array,
        the padding, and the size of the padded array
        will be printed.
    return_padded_array : bool
        If True, the padded array will be returned.
    
    Returns:
    -------
    small_4d_array : numpy array
        The smaller array which is the result of the reshaping.
    padding_return : tuple, optional
        The padding which was used to pad the array.

    """

    def resample_3d_array(big_array, volume_size):
        big_array_size = big_array.shape
        
        # Initialize an empty list to store padding values for each dimension
        padding = []

        # Iterate through each dimension (0, 1, and 2 for a 3D array)
        for i in range(3):
            # Calculate the remainder of the division of big_array_size[i] by volume_size[i]
            remainder = big_array_size[i] % volume_size[i]
            
            # Calculate the padding needed to make big_array_size[i] a multiple of volume_size[i]
            total_padding = volume_size[i] - remainder if remainder > 0 else 0
            
            # Divide the padding by 2 to distribute it equally on both sides of the array along that dimension
            padding_before = total_padding // 2
            padding_after = total_padding - padding_before

            # Append the padding value for the current dimension to the padding list
            padding.append((padding_before, padding_after))


        # Pad the big array
        padded_big_array = np.pad(big_array, padding, mode="constant")  # pad with zeros
        
        # Calculate the size of the small array
        small_3d_array_size = tuple(int(padded_big_array.shape[i] / volume_size[i]) for i in range(3))
            
        # Create a view of the bigger array with the desired volume size and the proper strides
        # note: The first three dimensions of strides represent the grid of the smaller 3D arrays, 
        # while the last three dimensions correspond to the volume_size of the resample cube.
        strides = tuple(padded_big_array.strides[k] * volume_size[k] for k in range(3)) + padded_big_array.strides
        window_view = np.lib.stride_tricks.as_strided(
            padded_big_array, shape=small_3d_array_size + volume_size, strides=strides
        )
        
        # Calculate the mean of each volume and reshape the result to the desired small array size
        small_3d_array = np.mean(window_view, axis=(-3, -2, -1))

        # throw an error if the following three shapes are not equal
        shape_testing_size = np.array(padded_big_array.shape) / np.array(volume_size)
        actual_small_3d_array_size = small_3d_array.shape
        if not (
            np.array_equal(shape_testing_size, small_3d_array_size)
            and np.array_equal(shape_testing_size, actual_small_3d_array_size)
            and np.array_equal(small_3d_array_size, actual_small_3d_array_size)
            ):
            raise ValueError("There is an error in the resample function!!")
        
        if print_array_sizes:
            print(f"Original array size: {big_array_size}",
                  f"Padding on both sides: {tuple(padding)}",
                  f"Padded array size: {padded_big_array.shape}",
                  f"Padded_big_array.shape / volume_size: {tuple(shape_testing_size)}",
                  f"Calculated small_3d_array_size: {small_3d_array_size}",
                  f"Effective small_3d_array_size: {small_3d_array.shape}",
                  sep="\n")
        
        return small_3d_array, padding
    

    ### Start of the resample_4d_array function ###

    # Create a tuple with the volume size
    volume_size = (reshape_cube, reshape_cube, reshape_cube)

    # call the function for each 3D array in the 4D array
    print("Resampling in progress...")
    small_4d_array = []
    for i in tqdm.tqdm(range(big_array_4d.shape[0])):
        # only print the array sizes for the first array
        if i > 0:
            print_array_sizes = False
        reshaped_array, padding_return = resample_3d_array(big_array_4d[i], volume_size)
        small_4d_array.append(reshaped_array)

    small_4d_array = np.array(small_4d_array)
    
    if return_padded_array:
        return small_4d_array, padding_return
    else:
        return small_4d_array