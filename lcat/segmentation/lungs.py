"""
Authors: Connor Brinton and Scotty Fleming
Segment lungs from a chest CT scan.
"""
import numpy as np
import skimage
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.segmentation

import lcat


def get_lung_segmentation(scan):
    """
    Given a `Scan` object representing a chest CT scan, return a binary mask representing the lungs
    (not including the air within the lungs).
    """
    # Identify filler pixels (top edge value)
    filler_value = get_top_value(np.concatenate((scan.voxels[[0, -1], :, :].flat,
                                                 scan.voxels[:, [0, -1], :].flat)))

    # Determine threshold
    scan_mask = np.logical_not(np.isclose(scan.voxels, filler_value))
    threshold = skimage.filters.threshold_otsu(scan.voxels[scan_mask])

    # Make sure threshold is within air/lung parameters
    if threshold <= -1000 or threshold >= 0:
        threshold = -500

    # Threshold the image
    foreground = scan.voxels >= threshold

    # Identify strongly connected components
    labels = skimage.measure.label(foreground, background=1, connectivity=1)

    # Remove edge components
    lcat.util.clear_border(labels, axis=[0, 1], in_place=True)

    # Identify the largest volume
    lung_mask = get_largest_volume(labels)

    # Fill edge holes by dilation
    # smoother = skimage.morphology.ball(10, dtype=bool)
    # lung_mask = skimage.morphology.binary_dilation(lung_mask, selem=smoother)

    smoother = skimage.morphology.disk(10, dtype=bool)
    for z_index in range(lung_mask.shape[-1]):
        lung_mask[..., z_index] = skimage.morphology.binary_dilation(lung_mask[..., z_index],
                                                                     selem=smoother)

    # Obtain lung envelope
    envelope_mask = get_lung_envelope(lung_mask)

    # Erode envelope to revert to proper extent
    # envelope_mask = skimage.morphology.binary_erosion(envelope_mask, selem=smoother)
    for z_index in range(lung_mask.shape[-1]):
        lung_mask[..., z_index] = skimage.morphology.binary_erosion(envelope_mask[..., z_index],
                                                                    selem=smoother)

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


def get_lung_envelope(lung_mask):
    """
    Given a mask representing thresholded lung values, obtain an envelope containing the lung region
    with no interior holes.
    """
    # Invert the mask
    reversed_mask = np.logical_not(lung_mask)

    # Identify connected_components
    reversed_labels = skimage.measure.label(reversed_mask)

    # Identify inner labels only
    inner_labels = lcat.util.clear_border(reversed_labels, axis=[0, 1])

    # Obtain only the outermost (edge-touching) region
    outside_mask = np.logical_xor(reversed_labels != 0, inner_labels != 0)

    # Invert the mask again to get the envelope
    envelope_mask = np.logical_not(outside_mask)

    return envelope_mask
