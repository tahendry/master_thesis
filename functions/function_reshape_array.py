"""
Author: Reto Hendry
Date: 2023-03-25
"""

import numpy as np

def reshape_array(big_array, volume_size):
    """
    Reshapes the bigger 3D array into a smaller 3D array by taking the mean of a volume
    from the bigger array and using the mean as one element of the smaller array.
    
    Args:
    big_array (ndarray): A 3D array representing the bigger array.
    volume_size (tuple): A tuple with 3 integers representing the size of the volume to create the mean.
    
    Returns:
    The reshaped smaller array with each element containing the mean of a corresponding volume
    from the bigger array.
    """

    big_array_size = big_array.shape
    
    # Calculate the padding needed to make the big array a multiple of the volume size
    padding = [((volume_size[i] - big_array_size[i] % volume_size[i]) % volume_size[i]) // 2 for i in range(3)]
    
    # Pad the big array
    padded_big_array = np.pad(big_array, [(padding[0], padding[0]), 
                                          (padding[1], padding[1]), 
                                          (padding[2], padding[2])])
    
    # Calculate the size of the small array
    small_array_size = tuple(int(padded_big_array.shape[i] / volume_size[i]) for i in range(3))
    small_array = np.zeros(small_array_size)
    
    # Iterate over each element of the smaller array
    for i in range(small_array_size[0]):
        for j in range(small_array_size[1]):
            for k in range(small_array_size[2]):
                # Calculate the mean of the volume from the bigger array
                volume_mean = np.mean(padded_big_array[i*volume_size[0]:(i+1)*volume_size[0],
                                                       j*volume_size[1]:(j+1)*volume_size[1],
                                                       k*volume_size[2]:(k+1)*volume_size[2]])
                # Assign the mean to the corresponding index in the smaller array
                small_array[i, j, k] = volume_mean
    
    return small_array