"""
Body depth featurization module.
"""
from __future__ import absolute_import

import numpy as np
import pandas as pd
import scipy.ndimage

import lcat
from . import registry


@registry.register_featurizer('body_depth')
def featurize_center(scan):
    """
    Featurize the given scan, returning body depth statistics.
    """
    # Create data distance placeholder
    index = pd.Index([], name='nodule_id')
    data = pd.DataFrame(index=index, columns=['min_body_depth',
                                              'mean_body_depth',
                                              'median_body_depth',
                                              'max_body_depth'])

    # Get the body segmentation for the scan
    body_shell = lcat.get_body_segmentation(scan)

    # Get body depths
    body_depths = scipy.ndimage.distance_transform_edt(body_shell)

    # Multiply by unit lengths (assume cubic unit cell)
    body_depths *= np.mean(scan.unit_cell[0])

    # For each nodule
    for nodule in scan.nodules:
        # Create full mask
        mask = lcat.util.get_full_nodule_mask(nodule, scan.voxels.shape)

        # Select tumor tracheal distances
        nodule_depths = body_depths[mask]

        # Add attributes to dataframe
        data.loc[nodule.nodule_id, :] = [
            np.min(nodule_depths),
            np.mean(nodule_depths),
            np.median(nodule_depths),
            np.max(nodule_depths)
        ]

    return data
