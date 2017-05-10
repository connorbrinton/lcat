#!/usr/bin/env python
"""
BMI 260: Final Project
Load segmentation for a chest CT scan from an xml format file.
"""
from collections import namedtuple
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import skimage
import skimage.measure
import skimage.segmentation

import preprocessing


# Scan datatype
Scan = namedtuple('Scan', ['voxels', 'nodules', 'unit_cell'])

# Nodule datatype
Nodule = namedtuple('Nodule', ['characteristics', 'mask'])

# XML namespace abbreviations
XMLNS = {
    'nih': 'http://www.nih.gov'
}

# Tag name regex
TAG_NAME_RE = re.compile('^{' + XMLNS['nih'] + '}' + '(.+)$')


def load_scan_data(scan_folder):
    """
    Loads the CT scan as a 3d voxel array, then loads the segmentation in the given dicom_folder by
    reading any xml files located in the folder and referencing dicom files as necessary. Returns a
    3D mask with the same dimensions as the CT scan.

    TODO: Combine nodules referring to the same entity. Unfortunately there is currently no unique
    ID for each nodule, meaning that multiple radiologist reads result in multiple almost-identical
    nodules in the scan metadata.
    """
    # Load the CT scan
    voxels, unit_cell, sop_instance_uids = preprocessing.load_folder(scan_folder)

    # Load all segmentations
    nodules = load_radiologist_annotations(scan_folder, voxels.shape, sop_instance_uids)

    # Convert to scan datatype
    scan = Scan(voxels, nodules, unit_cell)

    return scan


def load_radiologist_annotations(dicom_folder, dimensions, sop_instance_uids):
    """
    Load radiologist annotations (namely nodule characteristics and regions) from the xml files
    present in `dicom_folder`. Returns an array of Nodule objects representing all nodules found in
    the radiologist annotations.
    """
    # Create nodules placeholder
    nodules = []

    # Look for XML files
    for filename in os.listdir(dicom_folder):
        if filename.endswith('.xml'):
            # Reconstruct filepath
            filepath = os.path.join(dicom_folder, filename)

            # Load xml file
            tree = ET.parse(filepath)
            root = tree.getroot()

            # Find all nodules
            reads = root.findall('.//nih:readingSession//nih:unblindedReadNodule', XMLNS)

            # For each read
            for read in reads:
                # Extract nodule information
                nodules.append(get_nodule_information(read, dimensions, sop_instance_uids))

    return nodules


def get_nodule_information(read, dimensions, sop_instance_uids):
    """
    Given an unblindedReadNodule element, create a Nodule object representing the nodule's
    characteristics and vertices.
    """
    # Get characteristics
    characteristics = get_read_characteristics(read)

    # Get mask
    mask = get_read_mask(read, dimensions, sop_instance_uids)

    return Nodule(characteristics, mask)


def get_read_characteristics(read):
    """
    Get the characteristics from a read as recorded by the radiologist. Returns an empty dictionary
    if no characteristics were recorded.
    """
    # Extract characteristics
    characteristics = {}
    for attribute_elem in read.findall('.//nih:characteristics//*', XMLNS):
        # Get attribute name (removing namespace)
        match = TAG_NAME_RE.match(attribute_elem.tag)
        assert match is not None
        attribute_name = match.group(1)

        # Get attribute value
        attribute_value = int(attribute_elem.text)

        characteristics[attribute_name] = attribute_value

    return characteristics


def get_read_mask(read, dimensions, sop_instance_uids):
    """
    Get the vertices from a specific read.
    """
    # Create mask output placeholder
    mask = np.zeros(dimensions, dtype=bool)

    # Create holes queue
    holes = []

    # Identify regions of interest
    for roi_elem in read.findall('.//nih:roi', XMLNS):
        # Check if it's a hole
        if roi_elem.find('.//nih:inclusion', XMLNS).text.upper() == 'FALSE':
            holes.append(roi_elem)
        else:
            mark_region(mask, roi_elem, sop_instance_uids)

    # Create unincluded mask placeholder
    unincluded = np.zeros(dimensions, dtype=bool)

    # Identify hole regions
    for roi_elem in holes:
        mark_region(unincluded, roi_elem, sop_instance_uids)

    # Remove unincluded regions
    mask &= np.logical_not(unincluded)

    return mask


def mark_region(mask, roi_elem, sop_instance_uids):
    """
    Mark the region of interest encoded by `roi_elem` in `mask`. `sop_instance_uids` is used to
    determine the slices referenced by `roi_elem`.
    """
    # Create mask boundary placeholder
    mask_boundary = np.zeros(mask.shape[:2], dtype=bool)

    # Get Z index
    sop_instance_uid = roi_elem.find('.//nih:imageSOP_UID', XMLNS).text
    z_index = sop_instance_uids.index(sop_instance_uid)

    # Mark boundary points
    for edge_elem in roi_elem.findall('.//nih:edgeMap', XMLNS):
        # Get x and y positions
        x_position = int(edge_elem.find('.//nih:xCoord', XMLNS).text)
        y_position = int(edge_elem.find('.//nih:yCoord', XMLNS).text)

        # Mark boundary in mask
        mask_boundary[x_position, y_position] = 1

    # Fill in region
    mask_regions = skimage.measure.label(mask_boundary, background=-1, connectivity=1)
    mask_center = skimage.segmentation.clear_border(mask_regions)
    mask[:, :, z_index] |= mask_center != 0


def main():
    """
    Command-line invocation routine.
    """
    scan = load_scan_data('./data/LIDC-IDRI/LIDC-IDRI-0068/1.3.6.1.4.1.14519.5.2.1.6279.6001.709632090821449989953075380168/1.3.6.1.4.1.14519.5.2.1.6279.6001.187108608022306504546286626125')

    import IPython
    IPython.embed()


if __name__ == '__main__':
    main()
