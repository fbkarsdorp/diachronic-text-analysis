# Hierarchical Agglomerative Cluster Analysis
#
# Copyright (C) 2013 Folgert Karsdorp
# Author: Folgert Karsdorp <fbkarsdorp@gmail.com>
# URL: <https://github.com/fbkarsdorp/HAC-python>
# For licence information, see LICENCE.TXT

"""
Hierarchical Agglomerative Cluster Analysis.

This module consists of an implementation of Hierarchical Agglomerative
Cluster analysis (HAC), a method of cluster analysis in which items are grouped
together based on their similarity or dissimilarity in a bottom-up procedure. 
Besides the classical HAC analysis, this module gives a class of 
Variabilty-based Neighbor Clustering, a cluster method described in e.g. Hilpert 
& Gries (2006) in which the clustering procedure is temporarily constricted. 
In this cluster procedure, only clusters that are chronological neighbors are 
allowed to be clustered.

Both Clusterer and VNClusterer extend the AbstractClusterer interface which 
defines some common clustering operations, such as:
    1. cluster (cluster a sequence of vectors)
    2. iterate clusters (iterate the vectors in a specific order (for VNC))
    
Usage example (compare demo()):
    >>> from cluster.cluster import DistanceMatrix, Clusterer
    >>> from cluster.distance import euclidean_distance
    >>> vectors = [[2,4], [0,1], [1,1], [3,2], [4,0], [2,2]]
    >>> # compute the distance matrix on the basis of the vectors
    >>> dist_matrix = DistanceMatrix(vectors, euclidean_distance)
    >>> # plot the distance matrix
    >>> dist_matrix.draw()
    >>> # initialize a clusterer, with default linkage methode (Ward)
    >>> clusterer = Clusterer(dist_matrix)
    >>> # start the clustering procedure
    >>> clusterer.cluster(verbose=2)
    >>> # plot the result as a dendrogram
    >>> clusterer.dendrogram().draw(title=clusterer.linkage.__name__)
    
@author: Folgert Karsdorp
@requires: Python 2.6+
@version: 0.1
@license: GPL
@copyright: (c) 2011, Folgert Karsdorp
"""

from utils import *
from cluster import *
from dendrogram import *
from distance import *
from linkage import *

__all__ = ['Dendrogram', 'Clusterer', 'VNClusterer', 
           'CooccurrenceMatrix', 'DistanceMatrix']
