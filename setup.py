#!/usr/bin/env python
"""
Setup file for lcat-toolkit.
"""
from setuptools import setup

PACKAGES = ['lcat',
            'lcat.loading',
            'lcat.segmentation']

PACKAGE_DIR = {}

PACKAGE_DATA = {}

SCRIPTS = ['scripts/test_bronchi.py',
           'scripts/test_loading.py',
           'scripts/test_lung.py']

setup(name='lcat-toolkit',
      version='0.1.0',
      description='Lung Cancer Action Team (LCAT) toolkit',
      author='LCAT Team',
      author_email='lcat-toolkit@mail.stanford.edu',
      url='https://bitbucket.org/connorbrinton/bmi260-lcat-code',
      license='Unlicensed',
      package_data=PACKAGE_DATA,
      package_dir=PACKAGE_DIR,
      packages=PACKAGES,
      scripts=SCRIPTS)
