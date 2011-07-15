# Hierarchical Agglomerative Cluster Analysis
#
# Copyright (C) 2011 Institute for Dutch Lexicology (INL)
# Author: Folgert Karsdorp <folgert.karsdorp@inl.nl>
# URL: <http://www.inl.nl/>
# For licence information, see LICENCE.TXT

def _general_link(clusters, i, j, method):
    """
    This function is used to update the distance matrix in the clustering
    procedure.

    Several linkage methods for hierarchical agglomerative clustering
    can be used: single linkage, complete linkage, group average linkage,
    median linkage, centroid linkage and ward linkage.
    
    All linkage methods use the Lance-Williams update formula:
    d(ij,k) = a(i)*d(i,k) + a(j)*d(j,k) + b*d(i,j) + c*(d(i,k) - d(j,k))
    in the functions below, the following symbols represent the parameters in
    the update formula:
        n_x = length cluster
        a_x = a(x)
        d_xy = distance(x,y) or d(x,y)
    """
    for k in xrange(len(clusters)):
        if k != i and k != j:
            if method.__name__ == "ward_update":
                new_distance = method(clusters[i,k], clusters[j,k], k)
            else:
                new_distance = method(clusters[i,k], clusters[j,k])
            clusters[i,k] = new_distance
            clusters[k,i] = new_distance
    return clusters

def single_link(clusters, i, j, dendrogram):
    """
    Hierarchical Agglomerative Clustering using single linkage. Cluster j is
    clustered with cluster i when the minimum distance between any
    of the members of i and j is the smallest distance in the vector space.
    
    Lance-Williams parameters:
        a(i) = 0.5
        b = 0       =   min(d(i,k),d(j,k))
        c = -0.5
    """
    return _general_link(clusters, i, j, min)

def complete_link(clusters, i, j, dendrogram):
    """
    Hierarchical Agglomerative Clustering using complete linkage. Cluster j is
    clustered with cluster i when the maximum distance between any
    of the members of i and j is the smallest distance in the vector space.

    Lance-Williams parameters:
       a(i) = 0.5
       b = 0        =   max(d(i,k),d(j,k))   
       c = 0.5
    """
    return _general_link(clusters, i, j, max)

def average_link(clusters, i, j, dendrogram):
    """
    Hierarchical Agglomerative Clustering using group average linkage. Cluster j
    is clustered with cluster i when the pairwise average of values between the
    clusters is the smallest in the vector space.
    
    Lance-Williams parameters:
        a(i) = |i|/(|i|+|j|)
        b = 0
        c = 0    
    """
    n_i, n_j = len(dendrogram._items[i]), len(dendrogram._items[j])
    a_i = n_i / (n_i + n_j)
    a_j = n_j / (n_i + n_j)
    update_fn = lambda d_ik,d_jk: a_i*d_ik + a_j*d_jk
    return _general_link(clusters, i, j, update_fn)

def median_link(clusters, i, j, dendrogram):
    """
    Hierarchical Agglomerative Clustering using median linkage. Cluster j
    is clustered with cluster i when the distance between the median values
    of the clusters is the smallest in the vector space.
    
    Lance-Williams parameters:
        a(i) = 0.5
        b = -0.25
        c = 0
    """
    update_fn = lambda d_ik,d_jk: 0.5*d_ik + 0.5*d_jk + -0.25*clusters[i,j]
    return _general_link(clusters, i, j, update_fn)

def centroid_link(clusters, i, j, dendrogram):
    """
    Hierarchical Agglomerative Clustering using centroid linkage. Cluster j
    is clustered with cluster i when the distance between the centroids of the
    clusters is the smallest in the vector space.
    
    Lance-Williams parameters:
        a(i) = |i| / (|i| + |j|)
        b = -|i||j| / (|i|+ |j|)**2
        c = 0
    """
    n_i, n_j = len(dendrogram._items[i]), len(dendrogram._items[j])
    a_i = n_i / (n_i + n_j)
    a_j = n_j / (n_i + n_j)
    b = -(n_i * n_j) / (n_i + n_j)**2
    update_fn = lambda d_ik,d_jk: a_i*d_ik + a_j*d_jk + b*clusters[i,j]
    return _general_link(clusters, i, j, update_fn)

def ward_link(clusters, i, j, dendrogram):
    """
    Hierarchical Agglomerative Clustering using Ward's linkage. Two clusters i
    and j are merged when their merge results in the smallest increase in the
    sum of error squares in the vector space.
    
    Lance-Williams parameters:
        a(i) = (|i| + |k|) / (|i| + |j| + |k|)
        b = -|k|/(|i| + |j| + |k|)
        c = 0
    """
    n_i, n_j = len(dendrogram._items[i]), len(dendrogram._items[j])
    def ward_update(d_ik, d_jk, k):
        n_k = len(dendrogram._items[k])
        n_ijk = n_i+n_j+n_k
        return ( (n_i+n_k)/(n_ijk)*d_ik + (n_j+n_k)/(n_ijk)*d_jk +
                 -(n_k/(n_ijk))*clusters[i][j] )
    return _general_link(clusters, i, j, ward_update)

