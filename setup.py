#!/usr/bin/env python

from distutils.core import setup

setup(name = 'hacpy',
      version = '0.1',
      description = 'Hierarchical Agglomerative Cluster Analysis in Python',
      author = 'Folgert Karsdorp',
      author_email = 'folgert.karsdorp@inl.nl',
      py_modules = ['cluster', 'distance', 'linkage', 'api', 'corpus', 
                    'dendrogram', 'utils'])
