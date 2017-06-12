#!/usr/bin/env python
"""
BMI 260: Final Project
Bronchi segmentation/identification
"""
import sys

import numpy as np


def get_tracheal_distances(scan, lung_segmentation):
    """
    Calculate a distance map for every accessible voxel in the lung segmentation of the distance
    from the top of the trachea to the voxel. Returns a masked numpy array representing the distance
    assigned to each voxel.
    """
    # Lazy-load skfmm
    try:
        import skfmm
    except ImportError:
        print("Tracheal distance calculation requires scikit-fmm to be installed. Please execute "
              "`pip install scikit-fmm`.")
        sys.exit(1)

    # Obtain median threshold
    lung_threshold = np.median(scan.voxels[lung_segmentation])

    # Select non-air elements
    lung_tissue_mask = np.logical_and(lung_segmentation, scan.voxels > lung_threshold)

    # Compute boundaries affecting lung air
    air_boundaries = np.logical_or(lung_tissue_mask, np.logical_not(lung_segmentation))

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

    # Create same-size array of ones
    trachea_ceiling = np.ones(lung_segmentation.shape)

    # Set top of trachea as starting point
    trachea_ceiling[lung_segmentation[..., trachea_index]] = 0

    return trachea_ceiling
