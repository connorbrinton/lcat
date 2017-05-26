"""
Utility functions for the lcat toolkit.
"""
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy



def plot_slices(voxels, rows=6, columns=6, cmap=None):
    """
    Given a 3D array of voxels, plots slices.
    """
    # Create subplots
    figure, axes = plt.subplots(rows, columns, figsize=[13, 13])

    figure.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    # List each figure location
    product = list(itertools.product(range(rows), range(columns)))

    # Create appropriate slice indices
    indices = list(np.linspace(0, voxels.shape[-1] - 1, len(product), dtype=int).flat)

    # Loop over figures
    for index, (row, column) in zip(indices, product):
        # Get specific subaxes
        subaxes = axes[row, column]

        # Set title
        subaxes.set_title("Slice %d" % index)

        # Display slice
        subaxes.imshow(voxels[:, :, index], cmap=cmap, interpolation='none')

        # Remove axes
        subaxes.axis('off')

    # Show plot
    plt.show()


def save_slices(voxels, destination_folder, prefix="slice"):
    """
    Given a voxel array, save each slice as a tiff file with the name `prefix###.tiff` in the given
    `destination_folder`.
    """
    # Make the directory if necessary
    try:
        os.mkdir(destination_folder)
    except OSError:
        pass

    # Determine number of digits in largest slice number
    width = len(str(voxels.shape[-1]))

    # Generate format string
    filename_template = "%s%%0%dd.tiff" % (prefix, width)

    # Save each slice
    for z_index in range(voxels.shape[-1]):
        destination_filepath = os.path.join(destination_folder, filename_template % (z_index))
        scipy.misc.imsave(destination_filepath, voxels[..., z_index])
