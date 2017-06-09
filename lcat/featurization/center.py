"""
X, Y and Z featurization module.
"""
from __future__ import absolute_import

import pandas as pd
import scipy.ndimage

import lcat
from . import registry


@registry.register_featurizer('center')
def featurize_center(scan):
    """
    Featurize the given scan, returning tracheal distance statistics.
    """
    # Create data distance placeholder
    index = pd.Index([], name='nodule_id')
    data = pd.DataFrame(index=index, columns=['center_x', 'center_y', 'center_z'])

    # For each nodule
    for nodule in scan.nodules:
        # Create full mask
        mask = lcat.util.get_full_nodule_mask(nodule, scan.voxels.shape)

        # Calculate center of mass
        center = scipy.ndimage.measurements.center_of_mass(mask)

        # Convert to real space
        real_center = [coordinate * unit for coordinate, unit in zip(center, scan.unit_cell)]

        # Add attributes to dataframe
        data.loc[nodule.nodule_id] = real_center

    return data
