"""
Load data for a chest CT scan from both dicom and radiologist xml files.
"""
from collections import namedtuple

import lcat.loading.annotations
import lcat.loading.images


# Scan datatype
Scan = namedtuple('Scan', ['voxels', 'nodules', 'unit_cell'])


def load_scan(scan_folder):
    """
    Loads the CT scan as a 3d voxel array, then loads the segmentation in the given dicom_folder by
    reading any xml files located in the folder and referencing dicom files as necessary. Returns a
    3D mask with the same dimensions as the CT scan.

    TODO: Combine nodules referring to the same entity. Unfortunately there is currently no unique
    ID for each nodule, meaning that multiple radiologist reads result in multiple almost-identical
    nodules in the scan metadata.
    """
    # Load the CT scan
    voxels, unit_cell, sop_instance_uids = lcat.loading.images.load_folder(scan_folder)

    # Load all segmentations
    nodules = lcat.loading.annotations.load_radiologist_annotations(scan_folder, voxels.shape,
                                                                    sop_instance_uids)

    # Convert to scan datatype
    scan = Scan(voxels, nodules, unit_cell)

    return scan
