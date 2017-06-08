#!/usr/bin/env python
"""
Featurize tumor tracheal distance.
"""
from __future__ import print_function
import argparse
import os
import re

import numpy as np
import pandas as pd
import scipy.ndimage
from tqdm import tqdm

import lcat


DESCRIPTION = "Featurize tumor tracheal distance."

PATIENT_FOLDER_RE = re.compile('LIDC-IDRI-(.+)')


def execute(data_folder, destination_file):
    """
    Featurize tumor tracheal distance for all patients in `data_folder`, and write featurization to
    `destination_file`.
    """
    # Create data distance placeholder
    index = pd.MultiIndex(2 * [[]], 2 * [[]], names=['patient_id', 'nodule_id'])
    data = pd.DataFrame(index=index, columns=['min_tracheal_distance',
                                              'mean_tracheal_distance',
                                              'median_tracheal_distance',
                                              'max_tracheal_distance'])

    # List all patient paths
    patient_paths = list(generate_patient_paths(data_folder))

    # For each patient in the folder
    for patient_id, path in tqdm(patient_paths, desc="Featurizing"):
        # Update progress bar
        tqdm.write("Loading scan...")

        # Load scan
        scan = lcat.load_scan(path, cubify=True)

        # Update progress bar
        tqdm.write("Segmenting lung...")

        # Store scan features in data
        try:
            store_features(data, scan)
        except Exception as e:
            import IPython
            IPython.embed()

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


def store_features(data, scan):
    # Perform lung segmentation
    lung_segmentation = lcat.get_lung_segmentation(scan)

    # Update progress bar
    tqdm.write("Calculating tracheal distances...")

    # Get tracheal distances
    tracheal_distances = lcat.get_tracheal_distances(scan, lung_segmentation)

    # Fill unfilled entries
    # See https://stackoverflow.com/questions/3662361/
    indices = scipy.ndimage.distance_transform_edt(tracheal_distances.mask,
                                                   return_distances=False,
                                                   return_indices=True)
    tracheal_distances = tracheal_distances.data[tuple(indices)]

    # For each nodule
    for nodule in scan.nodules:
        # Create full mask
        mask = lcat.util.get_full_nodule_mask(nodule, scan.voxels.shape)

        # Select tumor tracheal distances
        tumor_distances = tracheal_distances[mask]

        # Add attributes to dataframe
        data.loc[(patient_id, nodule.nodule_id), :] = [
            np.min(tumor_distances),
            np.mean(tumor_distances),
            np.median(tumor_distances),
            np.max(tumor_distances)
        ]



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
