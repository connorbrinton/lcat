#!/usr/bin/env python
"""
BMI 260: Final Project
Load chest CT scan annotations from radiologist xml files.
"""
from collections import namedtuple
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import skimage
import skimage.measure
import skimage.segmentation

import lcat


# Nodule datatype
Nodule = namedtuple('Nodule', ['nodule_id', 'characteristics', 'origin', 'mask'])

# XML namespace abbreviations
XMLNS = {
    'nih': 'http://www.nih.gov'
}

# Tag name regex
TAG_NAME_RE = re.compile('^{' + XMLNS['nih'] + '}' + '(.+)$')


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
                nodule = get_nodule_information(read, dimensions, sop_instance_uids)

                # Only include >3mm nodules
                if any(dim > 1 for dim in nodule.mask.shape):
                    nodules.append(nodule)

    return nodules


def get_nodule_information(read, dimensions, sop_instance_uids):
    """
    Given an unblindedReadNodule element, create a Nodule object representing the nodule's
    characteristics and vertices.
    """
    # Get nodule ID
    nodule_id = get_read_nodule_id(read)

    # Get characteristics
    characteristics = get_read_characteristics(read)

    # Get mask
    origin, mask = get_read_mask(read, dimensions, sop_instance_uids)

    return Nodule(nodule_id, characteristics, origin, mask)


def get_read_nodule_id(read):
    # Find nodule ID element
    nodule_id_elem = read.find('.//nih:noduleID', XMLNS)

    # Return text content
    return nodule_id_elem.text


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
    Get a 3D array representing the region described by the specific read, prefaced by an origin
    specifying its placement in the image (in index coordinates).
    """
    # Get the full mask
    mask = get_mask_region(read, dimensions, sop_instance_uids)

    # Compress to small region with offset
    origin, mask = lcat.util.compress_nodule_mask(mask)

    return origin, mask


def get_mask_region(read, dimensions, sop_instance_uids):
    """
    Returns a full representation of the region represented by the given nodule read as a mask.
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
    import scans
    scan = scans.load_scan('../../data/LIDC-IDRI/LIDC-IDRI-0090')

    import IPython
    IPython.embed()


if __name__ == '__main__':
    main()
