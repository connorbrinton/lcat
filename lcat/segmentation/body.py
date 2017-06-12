"""
Segments a body from a CT scan.
"""
import numpy as np
import skimage
import skimage.measure
import skimage.segmentation

import lcat


def get_body_segmentation(scan):
    """
    Given a `Scan` object representing a chest CT scan, return a binary mask representing the region
    occupied by the body.
    """
    # Threshold the image (threshold is lower limit for lung tissue in HU)
    foreground = scan.voxels >= -700

    # Identify strongly connected components
    labels = skimage.measure.label(foreground, connectivity=1)

    # Identify the largest volume
    body_mask = get_largest_volume(labels)

    # Obtain body envelope
    envelope_mask = get_body_envelope(body_mask)

    return envelope_mask


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


def get_body_envelope(body_mask):
    """
    Given a mask representing thresholded lung values, obtain an envelope containing the lung region
    with no interior holes.
    """
    # Invert the mask
    reversed_mask = np.logical_not(body_mask)

    # Identify connected_components
    reversed_labels = skimage.measure.label(reversed_mask)

    # Identify inner labels only (on x and y edges, z outer labels can remain)
    inner_labels = lcat.util.clear_border(reversed_labels, axis=[0, 1])

    # Obtain only the outermost (edge-touching) region
    outside_mask = np.logical_xor(reversed_labels != 0, inner_labels != 0)

    # Invert the mask again to get the envelope
    envelope_mask = np.logical_not(outside_mask)

    return envelope_mask
