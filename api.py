# Hierarchical Agglomerative Cluster Analysis
#
# Copyright (C) 2011 Institute for Dutch Lexicology (INL)
# Author: Folgert Karsdorp <folgert.karsdorp@inl.nl>
# URL: <http://www.inl.nl/>
# For licence information, see LICENCE.TXT

class AbstractClusterer(object):
    """
    Abstract interface covering basic clustering functionality.
    """
    def iterate_clusters(self):
        """
        Iterate over all unique vector combinations in the matrix.
        """
        raise AssertionError('AbstractClusterer is an abstract interface')
        
    def smallest_distance(self, clusters):
        """
        Return the smallest distance in the distance matrix.
        The smallest distance depends on the possible connections in the
        distance matrix.
        """
        raise AssertionError('AbstractClusterer is an abstract interface')
        
    def cluster(self, verbose=0, sum_ess=False):
        """
        Cluster all clusters hierarchically unitl the level of 
        num_clusters is obtained.
        """
        raise AssertionError('AbstractClusterer is an abstract interface')
        
    def update_distmatrix(self, i, j, clusters):
        """
        Update the distance matrix using the specified linkage method, so that
        it represents the correct distances to the newly formed cluster.
        """
        return self.linkage(clusters, i, j, self._dendrogram)
        
    def dendrogram(self):
        """
        Return the dendrogram object.
        """
        return self._dendrogram
        
    def num_clusters(self):
        """
        Return the number of clusters.
        """
        return self._num_clusters

