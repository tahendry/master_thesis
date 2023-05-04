"""
Date: 04.05.2023
Author: Reto Hendry

"""

import numpy as np

def get_feature_positions(top_features, padding, resample_cube, shape_resampled_array):
    """
    Get the positions of the features in the original array.

    Parameters
    ----------
    top_features : list
        A list of integers containing the indices of the top features.
        i.e. [363, 196, 205, 357, 437, 259, 209, 277, 273, 124]
    padding : list
        A list of tuples containing the padding values for each dimension.
        i.e. [(2, 2), (0, 1), (2, 2)]
    resample_cube : integer
        The size of the resample cube.
    shape_resampled_array : tuple
        A tuple containing the shape of the resampled array.
        i.e. (19, 22, 19)

    Returns
    -------
    feature_positions : list
        A list of tuples (3d indices) containing the positions of the features in the original array.

    """

    def unpad_array(padded_array, padding):
        """
        Unpad a padded array.

        Parameters
        ----------
        padded_array : numpy.ndarray
            The padded array.
        padding : list
            A list of tuples containing the padding values for each dimension.

        Returns
        -------
        original_array : numpy.ndarray
            The original array before padding.
        
        """
        # Initialize an empty list to store the slices for each dimension
        slices = []
        
        # Loop through each dimension's padding values
        for pad in padding:
            start = pad[0]

            if pad[1] > 0:
                stop = -pad[1]
            else:
                stop = None  # slice till the end
            
            slices.append(slice(start, stop))
        
        original_array = padded_array[tuple(slices)]

        return original_array

    # create an array of np.nans with the shape of the resampled array
    marker_array = np.full(shape_resampled_array, np.nan)

    # Find the indices of the important elements in the unflattened array 
    # based on the flattened array
    top_indices_3d = [np.unravel_index(index, marker_array.shape) for index in top_features]
    for indx in top_indices_3d:
        marker_array[indx] = True

    # expand marker _array to the original shape
    expanded_marker_array = (marker_array
                            .repeat(resample_cube, axis=0)
                            .repeat(resample_cube, axis=1)
                            .repeat(resample_cube, axis=2)
    )

    # Reverse the padding operation
    original_marker_array = unpad_array(expanded_marker_array, padding)

    # Find the indices of the important elements in the 3d array
    marker = np.argwhere(original_marker_array == True)

    print("done with get_feature_positions")

    return marker