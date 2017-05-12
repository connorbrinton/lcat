"""
Authors: Connor Brinton and Scotty Fleming
Segment lungs from a chest CT scan.
"""
import numpy as np
import skimage
import skimage.filters
import skimage.measure
import skimage.segmentation


def get_lung_segmentation(scan):
    """
    Given a `Scan` object representing a chest CT scan, return a binary mask representing the lungs
    (not including the air within the lungs).
    """
    # Identify filler pixels (top edge value)
    filler_value = get_top_value(np.concatenate((scan.voxels[[0, -1], :, :].flat,
                                                 scan.voxels[:, [0, -1], :].flat)))

    # Determine threshold
    threshold = skimage.filters.threshold_otsu(scan.voxels[scan.voxels != filler_value])

    # Threshold the image
    foreground = scan.voxels >= threshold

    # Identify strongly connected components
    labels = skimage.measure.label(foreground, background=1, connectivity=1)

    # Remove edge components
    skimage.segmentation.clear_border(labels, in_place=True)

    # Identify the largest volume
    lung_mask = get_largest_volume(labels)

    return lung_mask


def get_largest_volume(labels):
    """
    Return a binary mask equivalent to the component with the largest volumes in the array `labels`.
    """
    return labels == get_top_value(labels[labels != 0])


def get_top_value(arr):
    """
    Given an ndarray, return the value which occurs most frequently.
    """
    # Count values
    values, counts = np.unique(arr, return_counts=True)

    # Identify top value
    top_value_index = np.argmax(counts)

    return values[top_value_index]
