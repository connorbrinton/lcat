"""
Authors: Connor Brinton and Scotty Fleming
Load a CT scan from a series of dicom files.
"""
import os

import dicom
import numpy as np


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

    # Get patient ID
    patient_id = get_single_value(dicom_object.PatientID for dicom_object in dicom_objects)

    # Obtain slice SOP instance UIDs
    sop_instance_uids = [dicom_object.SOPInstanceUID for dicom_object in dicom_objects]

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
    rescale_slopes = np.asarray([dicom_object.RescaleSlope for dicom_object in dicom_objects],
                                dtype=np.float32)
    rescale_intercepts = np.asarray([dicom_object.RescaleIntercept
                                     for dicom_object in dicom_objects], dtype=np.float32)
    rescaled_pixels = rescale_slopes[:, np.newaxis, np.newaxis] * np.asarray(pixel_arrays) \
                        + rescale_intercepts[:, np.newaxis, np.newaxis]

    # Roll pixels into xyz coordinate system
    voxels = np.moveaxis(rescaled_pixels, 0, -1)

    return patient_id, voxels, unit_cell, sop_instance_uids


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
