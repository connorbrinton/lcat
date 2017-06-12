#!/usr/bin/env python
"""
Setup file for lcat-toolkit.
"""
from setuptools import setup

PACKAGES = ['lcat',
            'lcat.analysis',
            'lcat.loading',
            'lcat.segmentation']

PACKAGE_DIR = {}

PACKAGE_DATA = {}

SCRIPTS = ['scripts/lcat-featurize.py',
           'scripts/lcat-visualize.py']

setup(name='lcat',
      version='0.1.0',
      description='Lung Cancer Action Team (LCAT) toolkit',
      author='LCAT Team',
      author_email='lcat-toolkit@mail.stanford.edu',
      url='https://github.com/connorbrinton/lcat',
      license='GPL-3.0',
      package_data=PACKAGE_DATA,
      package_dir=PACKAGE_DIR,
      packages=PACKAGES,
      scripts=SCRIPTS)
