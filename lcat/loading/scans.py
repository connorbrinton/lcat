"""
Load data for a chest CT scan from both dicom and radiologist xml files.
"""
from __future__ import division, print_function
from collections import namedtuple
import math
import warnings

import numpy as np
import scipy.ndimage

import lcat.loading.annotations
import lcat.loading.images


# Scan datatype
Scan = namedtuple('Scan', ['patient_id', 'voxels', 'nodules', 'unit_cell'])


def load_scan(scan_folder, cubify=False):
    """
    Loads the CT scan as a 3d voxel array, then loads the segmentation in the given dicom_folder by
    reading any xml files located in the folder and referencing dicom files as necessary. Returns a
    3D mask with the same dimensions as the CT scan.

    TODO: Combine nodules referring to the same entity. Unfortunately there is currently no unique
    ID for each nodule, meaning that multiple radiologist reads result in multiple almost-identical
    nodules in the scan metadata.
    """
    # Load the CT scan
    patient_id, voxels, unit_cell, sop_instance_uids = lcat.loading.images.load_folder(scan_folder)

    # Load all segmentations
    nodules = lcat.loading.annotations.load_radiologist_annotations(scan_folder, voxels.shape,
                                                                    sop_instance_uids)

    # Convert to scan datatype
    scan = Scan(patient_id, voxels, nodules, unit_cell)

    # Cubify if necessary
    if cubify:
        scan = cubify_scan(scan)

    return scan


def cubify_scan(scan):
    """
    Given a scan, interpolate the data to make the unit cell cubic. The dimension(s) with the
    smallest magnitude(s) in the unit cell will remain the same.
    """
    # Calculate scaling factors
    scaling_factors = get_scaling_factors(scan)

    # Calculate new unit cell size
    new_unit_cell = [step / factor for step, factor in zip(scan.unit_cell, scaling_factors)]

    # Rescale the voxels
    new_voxels = zoom_array(scan.voxels, scaling_factors)

    # Create nodules placeholder
    new_nodules = []

    # For each nodule
    for nodule in scan.nodules:
        # Perform nodule mask interpolation
        new_nodules.append(rescale_nodule(nodule, scaling_factors))

    # Return new Scan
    return Scan(scan.patient_id, new_voxels, new_nodules, new_unit_cell)


def get_scaling_factors(scan):
    """
    Returns the scaling factors for the given scan.
    """
    # Calculate the smallest step in unit_cell
    minimum_step = min(scan.unit_cell)

    # Calculate scaling factors
    scaling_factors = [step / minimum_step for step in scan.unit_cell]

    return scaling_factors


def zoom_array(array, factor, mode='nearest'):
    """
    Rescale the given `array` along each axis by `factor`. If `factor` is a sequence, each axis will
    be zoomed by its corresponding factor in `factor`.
    """
    with warnings.catch_warnings():
        # Filter warnings we can't fix
        warnings.filterwarnings("ignore", message="From scipy 0.13.0, the output shape of zoom() "
                                                  "is calculated with round() instead of int() - "
                                                  "for these inputs the size of the returned array "
                                                  "has changed.")

        # Return zoomed image
        return scipy.ndimage.zoom(array, factor, mode=mode)


def rescale_nodule(nodule, scaling_factors):
    """
    This interpolation process is trickier than the voxel interpolation process, because we're not
    only rescaling the mask, we're also moving the origin of the mask. In order to properly
    perform this interpolation, we perform the following steps:
    (1) Find the scaled origin as real numbers (not rounded)
    (2) Find the scaled anti-origin (opposite corner) as real numbers (not rounded)
    (3) Count the number of scaled cells along each dimension for mask
    (4) Convert rounded extents to original space
    (5) Perform rescaling by axis coordinates
    """
    # Load old extents
    old_starts = nodule.origin
    old_ends = [start + dim for start, dim in zip(old_starts, nodule.mask.shape)]

    # Calculate new extents
    new_starts = [start * factor for start, factor in zip(old_starts, scaling_factors)]
    new_ends = [end * factor for end, factor in zip(old_ends, scaling_factors)]

    # Calculate discrete extents
    discrete_starts = [int(math.floor(start)) for start in new_starts]
    discrete_ends = [int(math.ceil(end)) for end in new_ends]
    new_shape = [end - start for start, end in zip(discrete_starts, discrete_ends)]

    # Convert to original space
    unscaled_starts = [start / factor for start, factor in zip(discrete_starts, scaling_factors)]
    unscaled_ends = [end / factor for end, factor in zip(discrete_ends, scaling_factors)]

    # Generate interpolating points in original space
    axis_coordinates = [np.linspace(unscaled_start - start, unscaled_end - start, dim)
                        for start, unscaled_start, unscaled_end, dim
                        in zip(old_starts, unscaled_starts, unscaled_ends, new_shape)]

    # Perform interpolation
    new_mask = interpolate_array_by_axis(nodule.mask, axis_coordinates, output=float,
                                         mode='nearest')

    # Convert to binary array
    new_mask = new_mask > 0.5

    # Return rescaled nodule
    return lcat.loading.annotations.Nodule(nodule.nodule_id, nodule.characteristics,
                                           discrete_starts, new_mask)


def interpolate_array(array, new_shape, output=None, mode='nearest'):
    """
    Given an array and a target shape, perform an orthogonal transformation to the new shape using
    a spline interpolation.
    """
    # Generate mapping by axis
    axis_coordinates = [np.linspace(0, old_dim, new_dim)
                        for old_dim, new_dim in zip(array.shape, new_shape)]

    return interpolate_array_by_axis(array, axis_coordinates, output=output, mode=mode)


def interpolate_array_by_axis(array, axis_coordinates, output=None, mode='nearest'):
    # Expand axis coordinates
    coordinates = np.meshgrid(*axis_coordinates)

    # Perform interpolation
    # Note that mode only applies to input boundaries, the interpolation itself is a spline
    return scipy.ndimage.map_coordinates(array, coordinates, output=output, mode=mode)
