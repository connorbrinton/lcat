#!/usr/bin/env python
"""
Test scan loading given a particular directory. Once loaded, slices from the scan are displayed.
"""
# Import external dependencies
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

# Import our library
import lcat

# Load a patient's data
scan = lcat.load_scan('data/LIDC-IDRI-0090')

# Print the patient ID
print("Patient ID:", scan.patient_id)

# Display a slice of the scan
plt.imshow(scan.voxels[:, :, 30], interpolation='none')
plt.show()

# Print out the unit cell of the scan
print(scan.unit_cell)

# Get the first nodule
nodule = scan.nodules[0]

# Print out characteristics
print(nodule.characteristics)

# Get nodule mask reference
mask = nodule.mask

# Display the interesting slices of the nodule mask
interesting_indices = [z_index for z_index in range(mask.shape[-1]) if np.any(mask[:, :, z_index])]
for z_index in interesting_indices:
    plt.imshow(mask[:, :, z_index], interpolation='none')
    plt.show()
