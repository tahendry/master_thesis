"""
Date: 04.05.2023
Author: Reto Hendry

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_array_colored(array, marked_indices=None, linewidth_voxel=0):
    """
    Plot a 3D array with matplotlib.


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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(array.min(), array.max())
    
    filled = np.ones(array.shape, dtype=bool)
    facecolors = cmap(norm(array))
    
    if marked_indices is not None:
        marked_facecolors = np.copy(facecolors)
        marked_indices = np.array(marked_indices)
        marked_filled = np.zeros(array.shape, dtype=bool)
        marked_filled[tuple(marked_indices.T)] = True
        marked_facecolors[tuple(marked_indices.T)] = facecolors[tuple(marked_indices.T)]
        marked_facecolors[:,:,-1] = 0.8
        ax.voxels(marked_filled, facecolors=marked_facecolors, edgecolor='r', linewidth=1, alpha=None)
        
    ax.voxels(filled, facecolors=facecolors, edgecolor='k', linewidth=linewidth_voxel, alpha=0.6)
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.8)

    plt.show()