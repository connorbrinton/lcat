#!/usr/bin/env python
"""
Bronchi segmentation test harness.
"""
import argparse
import os

import lcat


DEFAULT_DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), '../data/redata/LIDC-IDRI-0068/')
DESCRIPTION = "Test lcat-toolkit segmentation code."


def test_bronchi_segmentation(scan_folder=DEFAULT_DATA_DIRECTORY):
    """
    Test bronchi segmentation
    """
    # Load scan
    scan = lcat.load_scan(scan_folder)

    # Perform lung segmentation
    lung_segmentation = lcat.get_lung_segmentation(scan)

    # Get bronchi segmentation
    bronchi_segmentation = lcat.get_bronchi_segmentation(lung_segmentation)


def main():
    """
    Launch bronchi segmentation test harness.
    """
    # Set up arguments
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('scan_folder', metavar="scan-folder", nargs='?',
                        default=DEFAULT_DATA_DIRECTORY,
                        help="Folder containing input dicom files of chest CT scans.")

    # Parse arguments
    args = parser.parse_args()

    # Test bronchi segmentation code
    test_bronchi_segmentation(args.scan_folder)


if __name__ == '__main__':
    main()