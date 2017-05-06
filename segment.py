#!/usr/bin/env python
"""
BMI 260: Homework 1
Authors: Connor Brinton and Scotty Fleming
Segment and display lungs from a chest CT scan.
"""
import argparse
import os

import dicom
import numpy as np
import skimage
import skimage.filters
import skimage.measure
import skimage.morphology
import stl


DESCRIPTION = "Segment lungs from a chest CT scan and write stl file representing surface."


def execute(dicom_folder, destination):
    """
    Load the dicom images from the given folder and segment out the lungs.
    """
    # Load voxels
    voxels, unit_cell = load_folder(dicom_folder)

    # Determine threshold
    threshold = skimage.filters.threshold_otsu(voxels[voxels != -3024])

    # Threshold the image
    foreground = voxels >= threshold

    # Identify strongly connected components
    labels = skimage.measure.label(foreground, background=1, connectivity=1)

    # Remove edge components
    remove_edge_components(labels)

    # Filter remaining components by volume
    apply_volume_filter(labels, 1000000)

    # Collapse components into single volume
    lung_mask = labels != 0

    # Perform a morphological closing to pick out the lighter colored blood vessels in the lung
    skimage.morphology.binary_closing(lung_mask, out=lung_mask)

    # Plot the volume surface
    write_stl_file(lung_mask, destination, spacing=unit_cell)


def load_folder(dicom_folder):
    """
    Given a folder of dicom files, load them and return a 3D numpy array representing the scan.
    Pixel values are converted to Houndsfield units using the rescaling slope and intercept encoded
    in each dicom file.
    """
    # Make sure folder exists
    if not os.path.isdir(dicom_folder):
        raise RuntimeError("The specified folder does not exist, or is a file.")

    # List all dicom files
    dicom_files = [filename for filename in os.listdir(dicom_folder) if filename.endswith(".dcm")]
    dicom_paths = [os.path.join(dicom_folder, filename) for filename in dicom_files]

    # Load each dicom object
    dicom_objects = [dicom.read_file(dicom_path) for dicom_path in dicom_paths]

    # Sort by instance number
    dicom_objects.sort(key=lambda x: x.InstanceNumber)

    # Extract pixel arrays
    pixel_arrays = [dicom_object.pixel_array for dicom_object in dicom_objects]

    # Extract physical slice dimensions
    pixel_spacing = get_single_value(dicom_object.PixelSpacing for dicom_object in dicom_objects)
    x_spacing, y_spacing = float(pixel_spacing[0]), float(pixel_spacing[1])

    # Extract slice separation distance
    slice_positions = [dicom_object.ImagePositionPatient[2] for dicom_object in dicom_objects]
    slice_separations = abs(np.diff(slice_positions))
    z_spacing = get_single_value(slice_separations)

    # Collapse physical dimensions into unit cell
    unit_cell = (x_spacing, y_spacing, z_spacing)

    # Rescale pixels
    rescale_slopes = np.asarray([dicom_object.RescaleSlope for dicom_object in dicom_objects])
    rescale_intercepts = np.asarray([dicom_object.RescaleIntercept
                                     for dicom_object in dicom_objects])
    rescaled_pixels = rescale_slopes[:, np.newaxis, np.newaxis] * np.asarray(pixel_arrays) \
                        + rescale_intercepts[:, np.newaxis, np.newaxis]

    # Roll pixels into xyz coordinate system
    voxels = np.moveaxis(rescaled_pixels, 0, -1)

    return voxels, unit_cell


def get_single_value(values):
    """
    Given a sequence of values, checks that all values are the same, and then returns the single
    value. The values must satisfy transitive equality.
    """
    # Attempt to obtain an iterator
    iterator = iter(values)

    # Get the initial value
    try:
        value = next(iterator)
    except StopIteration:
        raise ValueError("values cannot be of length zero.")

    # Make sure subsequent values match
    try:
        next_value = next(iterator)
        if value != next_value:
            raise ValueError("Not all elements in values are equal.")
    except StopIteration:
        pass

    # Everything matched, return value
    return value


def remove_edge_components(labels):
    """
    Remove components touching the edge of the image, setting them to zero.
    """
    # Identify components touching the edge
    edge_components = set(labels[0, :, :].flat) | set(labels[-1, :, :].flat) \
                        | set(labels[:, 0, :].flat) | set(labels[:, -1, :].flat)

    # Make components touching the edge background
    for edge_component in edge_components:
        labels[labels == edge_component] = 0


def apply_volume_filter(labels, threshold):
    """
    Apply a volume filter to the given components, setting components insufficiently large to zero.
    """
    # Identify remaining components
    components = np.unique(labels)

    # Apply volume filter
    for component in components:
        mask = labels == component
        if np.sum(mask) < threshold:
            labels[mask] = 0


def write_stl_file(volume, destination, spacing=None):
    """
    Create a 3D mesh using the marching cubes algorithm and generate an stl file, writing to the
    specified filepath.
    """
    # Provide default arguments
    if spacing is None:
        spacing = (1, 1, 1)

    # Obtain marching cubes surface
    vertices, faces, normals, _ = skimage.measure.marching_cubes(volume, spacing=spacing)

    # Create mesh data holder
    data = np.zeros(np.alen(faces), dtype=stl.Mesh.dtype)

    # Fill in vector data
    data['vectors'] = vertices[faces]

    # Fill in normal data. Each normal is the average of normals of the faces' three vertices.
    data['normals'] = normals[faces].mean(axis=1)
    mesh = stl.Mesh(data)

    # Save the mesh to an stl file
    mesh.save(destination)


def main():
    """
    Launch segmentation program.
    """
    # Set up arguments
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('dicom_folder', metavar="dicom-folder",
                        help="Folder containing input dicom files of chest CT scans.")
    parser.add_argument('stl_destination', metavar='stl-destination',
                        help="Destination for the output stl file. Overwrites existing files.")

    # Parse arguments
    args = parser.parse_args()

    # Execute program
    execute(args.dicom_folder, args.stl_destination)


if __name__ == '__main__':
    main()
