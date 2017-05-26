"""
Patient loading module of lcat. Defines the Patient class, which contains both a patient ID and a
scan.
"""
# System-level imports
from __future__ import absolute_import
from collections import namedtuple
import os
import re

# Local imports
import lcat.loading.scans


# Define Patient type
Patient = namedtuple('Patient', ['patient_id', 'scan'])

# Define folder regex
PATIENT_DIRECTORY_NAME_RE = re.compile(r'LIDC-IDRI-(\d{4})')


def load_patient(patient_directory):
    """
    Given a patient data directory, returns a Patient object. The Patient object includes the
    patient ID (inferred from the directory name) and the patient scan, loaded from the directory.
    """
    # Extract the patient directory name
    patient_directory_name = os.path.basename(os.path.normpath(patient_directory))

    # Perform regex matching
    patient_directory_name_match = PATIENT_DIRECTORY_NAME_RE.match(patient_directory_name)

    # Reject folder if no match found
    if patient_directory_name_match is None:
        raise Exception("Patient folder should be named LIDC-IDRI-####.")

    # Extract patient ID
    patient_id = patient_directory_name_match.group(1)

    # Load scan
    scan = lcat.loading.scans.load_scan(patient_directory)

    return Patient(patient_id, scan)
