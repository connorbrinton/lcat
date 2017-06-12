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


def get_bounding_box(arr):
    """
    Given an array of values, returns an list of tuples, where each tuple represents the extent of
    the non-zero values in `arr` along a particular axis.

    Inspired by http://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    """
    # Boundaries placeholder
    boundaries = []

    # For each axis
    for axis in range(arr.ndim):
        # Enumerate orthogonal axes
        other_axes = tuple(other_axis for other_axis in range(arr.ndim) if other_axis != axis)

        # Identify non-zero points
        nonzero = np.any(arr, axis=other_axes)

        # Identify non-zero region
        extent = tuple(np.where(nonzero)[0][[0, -1]])

        # Store boundary
        boundaries.append(extent)

    return boundaries


def crop_to_box(arr, box):
    """
    Given an array `arr` and boundaries `boundaries` (such as is returned by `get_bounding_box`),
    return `arr` cropped to `boundaries`.
    """
    # Calculate slices
    slicer = tuple(slice(extent[0], extent[1] + 1) for extent in box)

    # Perform slicing
    return arr[slicer]


def get_full_nodule_mask(nodule, scan_shape):
    # Create full placeholder
    mask = np.zeros(scan_shape, dtype=bool)

    # Create slicer
    slicer = tuple(slice(start, start + dim)
                   for start, dim in zip(nodule.origin, nodule.mask.shape))

    # Fill mask placeholder
    mask[slicer] = nodule.mask

    return mask


def compress_nodule_mask(mask):
    # Get bounding box
    bounding_box = get_bounding_box(mask)

    # Get origin from bounding box
    origin = [start for start, end in bounding_box]

    # Crop to bounding box
    mask = crop_to_box(mask, bounding_box)

    return origin, mask


def clear_border(labels, axis=None, in_place=False):
    """
    Clears any labeled components touching either border along the given axes (in `axis`).
    """
    # Convert labels to ndarray
    labels = np.asarray(labels)

    # Provide default axis argument
    if axis is None:
        axis = list(range(labels.ndim))

    # Make sure axis is 1d
    axes = np.atleast_1d(axis)
    assert axes.ndim == 1

    # Initialize to-be-cleared list
    touching = set()

    # For each axis
    for axis in axes.flat:
        # Generate selector
        selector = labels.ndim * [slice(None)]

        # Check leading border
        selector[axis] = 0
        touching |= set(np.unique(labels[tuple(selector)]))

        # Check trailing border
        selector[axis] = -1
        touching |= set(np.unique(labels[tuple(selector)]))

    if not in_place:
        labels = labels.copy()

    # Set matching values to zero
    for value in touching:
        labels[labels == value] = 0

    return labels


def image_from_mask(mask):
    """
    Convert a binary mask into a PIL image.
    """
    # Import PIL (if necessary)
    import PIL

    # Create mask image
    mask_image = PIL.Image.new('1', mask.shape)

    # Reserve memory for pixels
    pixels = mask_image.load()

    # Store pixel values
    for i in range(mask_image.size[0]):
        for j in range(mask_image.size[1]):
            pixels[i, j] = int(mask[i, j])

    return mask_image
