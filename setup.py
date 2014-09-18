#!/usr/bin/env python

import setuptools
from distutils.core import setup

setup(name = 'HACluster',
      version = '0.3',
      description = 'Hierarchical Agglomerative Cluster Analysis in Python',
      author = 'Folgert Karsdorp',
      author_email = 'fbkarsdorp@gmail.com',
      requires = ['numpy'],
      url = "https://github.com/fbkarsdorp/HAC-python",
      packages = ['HACluster'],
      platforms = 'Mac OS X, MS Windows, GNU Linux')
