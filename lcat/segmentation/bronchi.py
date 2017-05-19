#!/usr/bin/env python
"""
BMI 260: Final Project
Bronchi segmentation/identification
"""
import numpy as np
import skfmm
import skimage.morphology


def get_bronchi_segmentation(scan, lung_segmentation):
    # Obtain lung voxels
    lung_voxels = scan.voxels * lung_segmentation

    # Obtain threshold using Otsu's method
    # lung_threshold = skimage.filters.threshold_otsu(lung_voxels[lung_voxels != 0])
    lung_threshold = np.median(lung_voxels[lung_voxels != 0])

    # Select non-air elements
    lung_tissue = lung_voxels > lung_threshold

    # Compute boundaries affecting lung air
    air_boundaries = np.logical_or(lung_tissue, np.logical_not(lung_segmentation))

    # Choose seed point
    # TODO: Choose seed point better (actually find trachea, don't assume top)
    seed_boundary = get_seed_boundary(lung_segmentation)

    # Set up fast marching problem
    phi = np.ma.MaskedArray(seed_boundary, air_boundaries)

    # Perform fast marching problem
    distances = skfmm.distance(phi)

    # TODO: Return something else?
    return distances


def get_seed_boundary(lung_segmentation):
    """
    Given a `lung_segmentation`, identify and return the top slice of the trachea.
    """
    # Trachea disc placeholder
    trachea_index = None

    # For each slice
    for z_index in range(lung_segmentation.shape[2]):
        # Set candidate trachea disc
        candidate_trachea_disc = lung_segmentation[..., z_index]

        # Check for content
        if np.sum(candidate_trachea_disc) > 0:
            trachea_index = z_index
            break

    # Make sure there was a slice with content
    if trachea_index is None:
        raise Exception("No content in lung segmentation")

    # Create zero contour across top of trachea
    trachea_ceiling = np.ones(lung_segmentation.shape)
    trachea_ceiling[..., :(trachea_index + 1)] = -1

    return trachea_ceiling
