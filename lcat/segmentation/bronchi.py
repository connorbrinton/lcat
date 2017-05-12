#!/usr/bin/env python
"""
BMI 260: Final Project
Bronchi segmentation/identification
"""
import numpy as np
import skimage.morphology


def get_bronchi_segmentation(lung_segmentation):
    # Generate skeleton from lung segmentation
    skeleton = skimage.morphology.skeletonize_3d(np.logical_not(lung_segmentation))

    import IPython
    IPython.embed()
