"""
Characteristics featurization module.
"""
from __future__ import absolute_import

import numpy as np
import pandas as pd
import scipy.ndimage

import lcat
from . import registry


CHARACTERISTICS = [
	'subtlety',
	'internalStructure',
	'calcification',
	'sphericity',
	'margin',
	'lobulation',
	'spiculation',
	'texture',
	'malignancy',
]


@registry.register_featurizer('characteristics')
def featurize_characteristics(scan):
    """
    Featurize the given scan, returning nodule characteristics.
    """
    # Create data distance placeholder
    index = pd.Index([], name='nodule_id')
    data = pd.DataFrame(index=index, columns=CHARACTERISTICS)

    # For each nodule
    for nodule in scan.nodules:
        # Load characteristics
        data.loc[nodule.nodule_id] = [nodule.characteristics.get(attribute, np.nan)
                                      for attribute in CHARACTERISTICS]

    return data
