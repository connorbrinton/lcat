#!/usr/bin/env python
"""
Setup file for lcat-toolkit.
"""
from setuptools import setup

with open('requirements.txt', 'r') as source:
    DEPENDENCIES = [line.strip() for line in source.readlines()]

PACKAGES = ['lcat',
            'lcat.analysis',
            'lcat.loading',
            'lcat.segmentation']

PACKAGE_DIR = {}

PACKAGE_DATA = {}

SCRIPTS = ['scripts/lcat-featurize.py',
           'scripts/lcat-visualize.py']

setup(name='lcat',
      packages=PACKAGES,
      version='0.1.5',
      description='Lung Cancer Action Team (LCAT) toolkit',
      author='LCAT Team',
      author_email='brinton@cs.stanford.edu',
      url='https://github.com/connorbrinton/lcat',
      download_url='https://github.com/connorbrinton/lcat/archive/0.1.0.tar.gz',
      keywords=['CT', 'lung', 'biopsy'],
      license='GPL-3.0',
      package_data=PACKAGE_DATA,
      package_dir=PACKAGE_DIR,
      scripts=SCRIPTS,
      install_requires=DEPENDENCIES)
