#!/usr/bin/env python
"""
Generate frames for a tracheal distance algorithm video.
"""
from __future__ import division
import argparse
import os

import numpy as np
import scipy.ndimage
from tqdm import tqdm

import lcat


DESCRIPTION = "Visualize lcat-toolkit tracheal distance calculation code."

TRACHEAL_DISTANCE_FRAME_COUNT = 180
TRACHEAL_DISTANCE_RELATIVE_WINDOW = 2 * 1 / TRACHEAL_DISTANCE_FRAME_COUNT


def generate_movie(scan_folder, destination_folder):
    """
    Generate a movie visualizing tracheal distance calculation code.
    """
    # Load scan
    print("Loading scan...")
    scan = lcat.load_scan(scan_folder, cubify=False)

    # Write frames
    write_frames(scan, destination_folder)


def write_frames(scan, destination_folder):
    """
    Write frames of tracheal distance visualization movie.
    """
    # TODO: Write original scan frame(s)

    # Perform lung segmentation
    print("Segmenting lung...")
    lung_segmentation = lcat.get_lung_segmentation(scan)

    # TODO: Write lung segmentation frame(s)

    # Get tracheal distances
    print("Calculating tracheal distances...")
    tracheal_distances = lcat.get_tracheal_distances(scan, lung_segmentation)

    # Write tracheal distance frames
    write_tracheal_distance_frames(tracheal_distances, destination_folder)


def write_tracheal_distance_frames(tracheal_distances, destination_folder):
    """
    Write tracheal distances frame to the destination folder.
    """
    # Separate mask and data
    value_mask = np.logical_not(tracheal_distances.mask)
    tracheal_distances = tracheal_distances.data

    # Identify caclulated (non-zero) pixels
    distance_mask = np.logical_and(np.logical_not(np.isclose(tracheal_distances, 0)), value_mask)

    # Identify limits
    min_distance = np.min(tracheal_distances[distance_mask])
    max_distance = np.max(tracheal_distances[distance_mask])
    distance_range = max_distance - min_distance

    # Generate frame distances
    frame_distances = np.linspace(min_distance, max_distance, TRACHEAL_DISTANCE_FRAME_COUNT)

    # Loop over each frame
    for frame, distance in enumerate(tqdm(frame_distances.flat, desc="Writing frames")):
        # Calculate frame window
        absolute_window = distance_range * TRACHEAL_DISTANCE_RELATIVE_WINDOW
        window_start = max(min_distance, distance - absolute_window / 2)
        window_end = min(max_distance, distance + absolute_window / 2)

        # Generate mask
        frame_mask = np.logical_and(window_start <= tracheal_distances,
                                    tracheal_distances <= window_end)

        # Flatten mask
        frame_mask = np.max(frame_mask, axis=2).T

        # Determine output path
        filename = 'td-%03d.png' % frame
        filepath = os.path.join(destination_folder, filename)

        # Output frame
        lcat.util.image_from_mask(frame_mask).save(filepath)


def main():
    """
    Launch tracheal distance movie generator.
    """
    # Set up arguments
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('scan_folder', metavar="scan-folder",
                        help="Folder containing input dicom files of chest CT scans.")
    parser.add_argument('destination_folder', metavar="destination-folder",
                        help="Destination folder for slice .tiff files.")

    # Parse arguments
    args = parser.parse_args()

    # Generate tracheal distance movie
    generate_movie(args.scan_folder, args.destination_folder)


if __name__ == '__main__':
    main()
