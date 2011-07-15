# Hierarchical Agglomerative Cluster Analysis
#
# Copyright (C) 2011 Institute for Dutch Lexicology (INL)
# Author: Folgert Karsdorp <folgert.karsdorp@inl.nl>
# URL: <http://www.inl.nl/>
# For licence information, see LICENCE.TXT

import numpy
from numpy import dot sqrt

# distance measures
def cosine_distance(u, v):
    """Return the cosine distance between two vectors."""
    return 1.0 - dot(u, v) / (sqrt(dot(u, u)) * sqrt(dot(v, v)))

def euclidean_distance(u, v):
    """Return the euclidean distance between two vectors."""
    diff = u - v
    return sqrt(dot(diff, diff))

def cityblock_distance(u, v):
    """Return the Manhattan/City Block distance between two vectors."""
    return abs(u-v).sum()

def canberra_distance(u, v):
    """Return the canberra distance between two vectors."""
    return numpy.sum(abs(u-v) / abs(u+v))

def correlation(u, v):
    """Return the correlation distance between two vectors."""
    u_var = u - u.mean()
    v_var = v - v.mean()
    return 1.0 - dot(u_var, v_var) / (sqrt(dot(u_var, u_var)) *
                                      sqrt(dot(v_var, v_var)))
