"""
Feature registry module of Lung Cancer Action Team package.
"""
from __future__ import print_function
import itertools

import pandas as pd


# Featurizer placeholder
FEATURIZERS = {}


def register_featurizer(featurizer_name):
    """
    Function decorator which registers the given function under the argument `featurizer_name` as
    a featurizer. The function must accept a single scan and return a pandas DataFrame containing
    columns representing features and rows representing nodules. The Index must specify the
    nodule_id for each nodule.
    """
    def decorator(featurize):
        """
        Register the function `featurize` as a featurizer
        """
        # Store the function
        FEATURIZERS[featurizer_name] = featurize

        # Return it unchanged
        return featurize

        # Return a function decorator to register featurizer
    return decorator


def featurize_scan(scan):
    """
    Featurize the given scan, using all available featurizers. Returns a pandas DataFrame with all
    featurizer results, indexed by patient_id and nodule_id using a MultiIndex.
    """
    # Featurizations placeholder
    featurizations = []

    # For each featurizer
    for featurizer_name, featurizer in FEATURIZERS.iteritems():
        # Featurize the scan
        try:
            featurizations.append(featurizer(scan))
        except:
            print("Error featurizing scan for patient '%s' using featurizer '%s', skipping..."
                  % (scan.patient_id, featurizer_name))

    # Concatenate all featurizations
    return pd.concat(featurizations, axis=1)


def featurize_scan_single(scan, featurizer_name):
    """
    Featurize the given scan, using the specified featurizer. Returns a pandas DataFrame with all
    featurizer results, indexed by patient_id and nodule_id using a MultiIndex.
    """
    # Get the featurizer
    featurizer = FEATURIZERS[featurizer_name]

    # Featurize the scan
    featurization = featurizer(scan)

    return featurization
