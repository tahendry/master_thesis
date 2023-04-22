"""
Author: Reto Hendry
Date: 2023-03-25
"""

import numpy as np
import tqdm

def resample_4d_array(big_array_4d, reshape_cube):
    """
    This code defines a function that resamples a 3D NumPy array 
    by taking the mean of non-overlapping volumes of 
    a specified size. The input array is first padded to ensure that 
    it can be evenly divided into the smaller volumes.
    
    Parameters:
    ----------
    big_array_4d : numpy array
        The bigger array which is to be reshaped.
    reshape_cube : int
        The size of the volume (SxSxS) which 
        is to be used to calculate the mean
    
    Returns:
    -------
    small_4d_array : numpy array
        The smaller array which is the result of the reshaping.

    """

    def resample_3d_array(big_array, volume_size):
        big_array_size = big_array.shape
        
        # Calculate the padding needed to make the big array a multiple of the volume size
        padding = [((volume_size[i] - big_array_size[i] % volume_size[i]) % volume_size[i]) // 2 for i in range(3)]
        
        # Pad the big array
        padded_big_array = np.pad(big_array, [(padding[0], padding[0]), 
                                            (padding[1], padding[1]), 
                                            (padding[2], padding[2])])
        
        # Calculate the size of the small array
        small_4d_array_size = tuple(int(padded_big_array.shape[i] / volume_size[i]) for i in range(3))
        
        # Create a view of the bigger array with the desired volume size and the proper strides
        strides = tuple(padded_big_array.strides[i] * volume_size[i] for i in range(3)) + padded_big_array.strides
        window_view = np.lib.stride_tricks.as_strided(padded_big_array, shape=small_4d_array_size + volume_size, strides=strides)
        
        # Calculate the mean of each volume and reshape the result to the desired small array size
        small_4d_array = np.mean(window_view, axis=(-3, -2, -1)).reshape(small_4d_array_size)
        
        return small_4d_array
    
    # Create a tuple with the volume size
    volume_size = (reshape_cube, reshape_cube, reshape_cube)

    # call the function
    print("Resampling in progress...")
    small_4d_array = []
    for i in tqdm.tqdm(range(big_array_4d.shape[0])):
        reshaped_array = resample_3d_array(big_array_4d[i], volume_size)
        small_4d_array.append(reshaped_array)

    small_4d_array = np.array(small_4d_array)
    

    return small_4d_array