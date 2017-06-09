"""
X, Y and Z featurization module.
"""
from __future__ import absolute_import, division

import numpy as np
import pandas as pd
import scipy.ndimage
import skimage.measure

import lcat
from . import registry


REGION_PROPERTIES = [
    'volume',
    'equivalent_diameter',
    # 'major_axis_length',
    # 'minor_axis_length',
    'min_intensity',
    'mean_intensity',
    'max_intensity',
]


@registry.register_featurizer('region_properties')
def featurize_region_properties(scan):
    """
    Featurize the given scan, returning tracheal distance statistics.
    """
    # Create data distance placeholder
    index = pd.Index([], name='nodule_id')
    data = pd.DataFrame(index=index, columns=REGION_PROPERTIES)

    # For each nodule
    for nodule in scan.nodules:
        # Create full mask
        mask = lcat.util.get_full_nodule_mask(nodule, scan.voxels.shape)

        # Add attributes to dataframe
        data.loc[nodule.nodule_id] = [
            calculate_volume(mask, scan.unit_cell),
            calculate_equivalent_diameter(mask, scan.unit_cell),
            calculate_min_intensity(mask, scan.voxels),
            calculate_mean_intensity(mask, scan.voxels),
            calculate_max_intensity(mask, scan.voxels),
        ]

    return data


def calculate_volume(nodule_mask, unit_cell):
    """
    Calculate and return the volume occupied by the nodule represented by the given nodule_mask.
    """
    # Calculate unit cell volume
    unit_volume = np.prod(unit_cell)

    # Calculate volume occupied
    return unit_volume * nodule_mask.sum()


def calculate_equivalent_diameter(nodule_mask, unit_cell):
    """
    Calculate and return the equivalent diameter of a sphere with the same volume as the given
    nodule.
    """
    # Calculate volume
    volume = calculate_volume(nodule_mask, unit_cell)

    return (3 * volume / (4 * np.pi)) ** (1 / 3)


def calculate_min_intensity(nodule_mask, intensity_image):
    """
    Calculate and return the minimum intensity of the nodule represented by `nodule_mask` in
    `intensity_image`
    """
    return np.min(intensity_image[nodule_mask])


def calculate_mean_intensity(nodule_mask, intensity_image):
    """
    Calculate and return the average intensity of the nodule represented by `nodule_mask` in
    `intensity_image`
    """
    return np.mean(intensity_image[nodule_mask])


def calculate_max_intensity(nodule_mask, intensity_image):
    """
    Calculate and return the maximum intensity of the nodule represented by `nodule_mask` in
    `intensity_image`
    """
    return np.max(intensity_image[nodule_mask])
