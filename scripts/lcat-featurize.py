#!/usr/bin/env python
"""
Featurize tumor tracheal distance.
"""
from __future__ import print_function
import argparse
import itertools
import os
import re

import pandas as pd
from tqdm import tqdm

import lcat
import lcat.featurization


DESCRIPTION = "Featurize tumor tracheal distance."

PATIENT_FOLDER_RE = re.compile('LIDC-IDRI-(.+)')


def execute(data_folder, destination_file):
    """
    Featurize tumor tracheal distance for all patients in `data_folder`, and write featurization to
    `destination_file`.
    """
    # Create featurizations placeholder
    featurizations = []

    # List all patient paths
    patient_paths = list(generate_patient_paths(data_folder))

    # For each patient in the folder
    for patient_id, path in tqdm(patient_paths, desc="Featurizing"):
        # Update progress bar
        tqdm.write("Loading scan for patient %s..." % patient_id)

        # Load scan
        scan = lcat.load_scan(path, cubify=True)

        tqdm.write("Featurizing scan...")

        # Featurize scan
        featurization = lcat.featurization.featurize_scan(scan)

        if len(featurization) == 0:
            continue

        # Generate new index content
        index_content = zip(itertools.repeat(scan.patient_id), featurization.index)

        # Add the patient id column
        index = pd.MultiIndex.from_tuples(index_content, names=['patient_id', 'nodule_id'])

        # Apply new index
        featurization.index = index

        # Store featurization for scan
        featurizations.append(featurization)

    # Combine featurizations
    data = pd.concat(featurizations)

    # Write data to file
    data.to_csv(destination_file)


def generate_patient_paths(data_folder):
    """
    Yield a patient ID and path for each patient folder in data_folder.
    """
    # For every filename in data_folder
    for filename in os.listdir(data_folder):
        # Generate full path
        filepath = os.path.join(data_folder, filename)

        # Make sure it's a folder
        if not os.path.isdir(filepath):
            continue

        # Perform regex matching
        match = PATIENT_FOLDER_RE.match(filename)

        # Reject improperly named folders
        if match is None:
            continue

        # Yield patient ID and path
        yield match.group(1), filepath


def main():
    """
    Launch bronchi segmentation test harness.
    """
    # Set up arguments
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('data_folder', metavar="scan-folder",
                        help="Folder containing LIDC-IDRI data.")
    parser.add_argument('destination_file', metavar="destination-file",
                        help="Destination CSV file for featurization.")

    # Parse arguments
    args = parser.parse_args()

    # Test bronchi segmentation code
    execute(args.data_folder, args.destination_file)


if __name__ == '__main__':
    main()
