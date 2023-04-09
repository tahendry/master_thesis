"""
Author: Reto Hendry
Date: 2023-03-25
"""

import numpy as np

def reshape_array(big_array_4d, reshape_cube):
    """
    Reshapes the bigger 3D array into a smaller 3D array by taking the mean of a volume
    from the bigger array and using the mean as one element of the smaller array.
    
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

    def reshape_3d_array(big_array, volume_size):
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
    
    # Create a tuple with the volume size
    volume_size = (reshape_cube, reshape_cube, reshape_cube)

    # call the function
    small_4d_array = np.array(
        [reshape_3d_array(big_array_4d[i], volume_size) 
            for i in range(big_array_4d.shape[0])]
    )
    
    return small_4d_array