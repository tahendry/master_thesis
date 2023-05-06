"""
Date: 04.05.2023
Author: Reto Hendry

"""

import numpy as np
import matplotlib.pyplot as plt

def plot_3d_array_colored(array, marked_indices=None, linewidth_voxel=0):
    """
    Plot a 3D array with matplotlib. The voxels which have a value of 0 are transparent.
    The voxels which have a value > 0 are colored according to their value.
    The voxels witch are marked via the marked_indices argument, have a red edge.


    Parameters
    ----------
    array : 3D-array [x, y, z]
        The array with the data.
    marked_indices : list, optional
        List with the indices of the voxels which should be marked, by default None.
    linewidth_voxel : int, optional
        The linewidth of the voxels which are not marked, by default 0.
        It helps to see the edges of the reversed resampled array.

    Return
    ------
    None -> plot the array

    """

    print("creating 3D plot...")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(array.min(), array.max())
    
    filled = (array != 0)  # Update this line to consider only non-zero values
    facecolors = cmap(norm(array))
    
    # Set alpha channel to 0 where array values are 0
    facecolors[array == 0, -1] = 0.0

    # Set the alpha value for non-transparent voxels
    facecolors[array != 0, -1] = 0.6

    
    if marked_indices is not None:
        marked_facecolors = np.copy(facecolors)
        marked_indices = np.array(marked_indices)
        marked_filled = np.zeros(array.shape, dtype=bool)
        marked_filled[tuple(marked_indices.T)] = True
        marked_facecolors[tuple(marked_indices.T)] = facecolors[tuple(marked_indices.T)]
        marked_facecolors[:,:,-1] = 0.8
        ax.voxels(marked_filled, facecolors=marked_facecolors, edgecolor='r', linewidth=1, alpha=None)
        
    ax.voxels(filled, facecolors=facecolors, edgecolor='k', linewidth=linewidth_voxel, alpha=None)
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    ax.view_init(elev=25, azim=-110)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.8)
    plt.tight_layout()

    plt.show()