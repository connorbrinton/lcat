"""
Tracheal distance featurization module.
"""
from __future__ import absolute_import

import numpy as np
import pandas as pd
import scipy.ndimage

import lcat
from . import registry


@registry.register_featurizer('tracheal_distance')
def featurize_tracheal_distance(scan):
    """
    Featurize the given scan, returning tracheal distance statistics.
    """
    # Create data distance placeholder
    index = pd.Index([], name='nodule_id')
    data = pd.DataFrame(index=index, columns=['min_tracheal_distance',
                                              'mean_tracheal_distance',
                                              'median_tracheal_distance',
                                              'max_tracheal_distance'])

    # Perform lung segmentation
    lung_segmentation = lcat.get_lung_segmentation(scan)

    # Get tracheal distances
    tracheal_distances = lcat.get_tracheal_distances(scan, lung_segmentation)

    # Fill unfilled entries
    # See https://stackoverflow.com/questions/3662361/
    indices = scipy.ndimage.distance_transform_edt(tracheal_distances.mask,
                                                   return_distances=False,
                                                   return_indices=True)
    tracheal_distances = tracheal_distances.data[tuple(indices)]

    # For each nodule
    for nodule in scan.nodules:
        # Create full mask
        mask = lcat.util.get_full_nodule_mask(nodule, scan.voxels.shape)

        # Select tumor tracheal distances
        tumor_distances = tracheal_distances[mask]

        # Add attributes to dataframe
        data.loc[nodule.nodule_id, :] = [
            np.min(tumor_distances),
            np.mean(tumor_distances),
            np.median(tumor_distances),
            np.max(tumor_distances)
        ]

    return data
