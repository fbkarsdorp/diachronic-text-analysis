# Hierarchical Agglomerative Cluster Analysis
#
# Copyright (C) 2011 Institute for Dutch Lexicology (INL)
# Author: Folgert Karsdorp <folgert.karsdorp@inl.nl>
# URL: <http://www.inl.nl/>
# For licence information, see LICENCE.TXT

from __future__ import division

import numpy
import copy
import argparse
import codecs
import os
import re
import string

# TODO, someday, implement my own plotting function of the dendrogram.
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
from numpy import log, sqrt, dot
from operator import itemgetter
from collections import defaultdict
from itertools import combinations


def ngrams(text, n):
    """Return N grams of a text."""
    count = max(0, len(text) - n + 1)
    return (tuple(text[i:i+n]) for i in xrange(count))

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


# TODO MOVE TO OTHER FILE? (SPECIFIC CLASS FOR GTB-CORPUS)
class Text(object):
    def __init__(self, id, title, author, year, text, kind=None):
        self._id = id
        self._text = text
        self._title = title
        self._author = author
        self._year = int(year)
        self._kind = kind
        # remove all punctuation, transform to lowercase and split on spaces
        self._tokens = text.translate(None, string.punctuation).lower().split()

    def __hash__(self): # IS THIS ALLOWED?
        return self._id

    def title(self): return self._title
    def author(self): return self._author
    def year(self): return self._year
    def kind(self): return self._kind

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self._tokens[i.start:i.stop]
        else:
            return self._tokens[i]

    def __len__(self):
        return len(self._tokens)


class CooccurrenceMatrix(numpy.ndarray):
    """ Represents a co-occurrence matrix. """
    def __new__(cls, data, dtype=None):
        if not isinstance(data, CooccurrenceMatrix):
            data, rownames, colnames = CooccurrenceMatrix.convert(data)
        else:
            rownames, colnames = data.rownames, data.colnames
        obj = numpy.asarray(data).view(cls)
        obj.rownames = rownames
        obj.colnames = colnames
        return obj

    def __array_finialize__(self, obj):
        if obj is None: return
        self.rownames = getattr(obj, 'rownames', None)
        self.colnames = getattr(obj, 'colnames', None)

    @classmethod
    def convert(cls, data):
        matrix = numpy.zeros((len(set(k for k,v in data)),
                             len(set(v for k,v in data))))
        colnames, rownames = {}, {}
        for k,v in sorted(data):
            if k not in rownames:
                rownames[k] = len(rownames)
            if v not in colnames:
                colnames[v] = len(colnames)
            matrix[rownames[k],colnames[v]] += 1
        rownames = [k for k,v in sorted(rownames.items(), key=itemgetter(1))]
        colnames = [k for k,v in sorted(colnames.items(), key=itemgetter(1))]
        return matrix, rownames, colnames

    def tfidf(self):
        """
        Returns a matrix in which for all entries in the co-occurence matrix
        the 'term frequency-inverse document frequency' is calculated.
        """
        matrix = numpy.zeros(self.shape)
        # the number of words in a document
        words_per_doc = self.sum(axis=1)
        # the number of documents in which a word is attested.
        word_frequencies = numpy.sum(self > 0, axis=0)
        # calculate the term frequencies
        for i in xrange(self.shape[0]):
            tf = self[i] / words_per_doc[i] # array of tf's
            matrix[i] = tf * (log(self.shape[0] / word_frequencies))
        return matrix


class DistanceMatrix(numpy.ndarray):
    """
    Simple wrapper around numpy.ndarray, to provide some custom
    Distance Matrix functionality like plotting the distance matrix
    with matplotlib.
    """
    def __new__(cls, input_array, dist_metric=euclidean_distance, lower=True):
        if (not isinstance(input_array, (numpy.ndarray, DistanceMatrix))
            or len(input_array) != len(input_array[0])
            or not max(numpy.diag(input_array)) == 0):
            input_array = DistanceMatrix.convert_to_distmatrix(
                numpy.array(input_array), dist_metric, lower=lower)
        obj = numpy.asarray(input_array).view(cls)
        obj.distance_metric = dist_metric
        return obj

    def __array_finialize__(self, obj):
        if obj is None: return
        self.distance_metric = getattr(obj, 'distance_metric', None)

    @classmethod
    def convert_to_distmatrix(cls, data, distance, lower=True):
        matrix = numpy.zeros((len(data), len(data)))
        for i,j in combinations(xrange(len(data)), 2):
            matrix[i][j] = distance(data[i], data[j])
            if lower == True:
                matrix[j][i] = matrix[i][j]
        # add a nan-diagonal, useful for further computations.
        numpy.fill_diagonal(matrix, numpy.nan)
        return matrix

    def diag_is_zero(self):
        """Check if the diagonal contains only distances of 0."""
        return max(numpy.diag(self)) == 0

    def remove(self, idx):
        """
        Delete a row and column with index IDX.
        WARNING this function is NOT destructive!
        """
        indices = range(len(self))
        indices.remove(idx)
        return self.take(indices, axis=0).take(indices, axis=1)

    def draw(self, save=False, format="pdf"):
        """Make a nice colorful plot of the distance matrix."""
        try:
            import pylab
        except ImportError:
            raise ImportError("Install pylab.")
        fig = pylab.figure()
        axmatrix = fig.add_axes([0.1,0.1,0.8,0.8])
        im = axmatrix.matshow(self, aspect='auto', origin='upper',
                              cmap=pylab.cm.YlGnBu)
        axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.8])
        pylab.colorbar(im, cax=axcolor)
        fig.show()
        if save:
            fig.savefig('distance-matrix.%s' % (format,))

    def summary(self):
        """Return a small summary of the matrix."""
        print 'DistanceMatrix (n=%s)' % len(self)
        print 'Distance metric = %s' % self.distance_metric.__name__
        print self


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


class DendrogramNode(object):
    """Represents a node in a dendrogram."""
    def __init__(self, id, *children):
        self.id = id
        self.distance = 0.0
        self._children = children

    def leaves(self):
        """Return the leaves of all children of a given node."""
        if self._children:
            leaves = []
            for child in self._children:
                leaves.extend(child.leaves())
            return leaves
        else:
            return [self]

    def adjacency_list(self):
        """
        For each merge in the dendrogram, return the direct children of
        the cluster, the distance between them and the number of items in the
        cluster (the total number of children all the way down the tree).
        """
        if self._children:
            a_list = [(self._children[0].id, self._children[1].id,
                       self.distance, len(self))]
            for child in self._children:
                a_list.extend(child.adjacency_list())
            return a_list
        else: return []

    def __len__(self):
        return len(self.leaves())


class Dendrogram(object):
    """
    Class representing a dendrogram. Part is inspired by the Dendrogram class
    of NLTK. It is adjusted to work properly and more efficiently with
    matplotlib and VNC. 
    """
    def __init__(self, items):
        self._items = [DendrogramNode(i) for i in xrange(len(items))]
        self._original_items = copy.copy(self._items)
        self._num_items = len(self._items)

    def merge(self, *indices):
        """
        Merge two or more nodes at the given INDICES in the dendrogram.
        The new node will get the index of the first node specified.
        """
        assert len(indices) >= 2
        node = DendrogramNode(
            self._num_items, *[self._items[i] for i in indices])
        self._num_items += 1
        self._items[indices[0]] = node
        for i in indices[1:]:
            del self._items[i]

    def draw(self, show=True, save=False, format="pdf", labels=None, title=None):
        """Draw the dendrogram using pylab and matplotlib."""
        try:
            import pylab
        except ImportError:
            raise ImportError("Pylab not installed, can't draw dendrogram")
        fig = pylab.figure()
        m = numpy.array(sorted(self._items[0].adjacency_list(), key=itemgetter(2)),
                  numpy.dtype('d'))
        # default labels are the cluster id's (these must be matched!!)
        d = scipy_dendrogram(m, labels=labels, color_threshold=0.6*max(m[:,2]))
        if title is not None:
            fig.suptitle(title, fontsize=12)
        if show:
            fig.show()
        if save:
            fig.savefig('dendrogram.%s' % (format,))


class Clusterer(object):
    """
    The Hierarchical Agglomerative Clusterer starts with each of the N vectors
    as singleton clusters. It then iteratively merges pairs of clusters which
    have the smallest distance according to function LINKAGE. This continues
    until there is only one cluster.
    """
    def __init__(self, data, dist_metric=euclidean_distance,
                 linkage = ward_link, num_clusters=1):
        self._num_clusters = num_clusters
        if isinstance(data, DistanceMatrix):
            vector_ids = [[i] for i in range(len(data))]
        elif isinstance(data, (numpy.ndarray, list)):
            vector_ids = data
            data = DistanceMatrix(data, dist_metric)
        else:
            raise ValueError('Input must by of type list or DistanceMatrix')
        self._dendrogram = Dendrogram(vector_ids)
        self._dist_matrix = data
        self.linkage = linkage

    def iterate_clusters(self):
        """Iterate over all unique vector combinations in the matrix."""
        raise NotImplementedError()

    def smallest_distance(self, clusters):
        """
        Return the smallest distance in the distance matrix.
        The smallest distance depends on the possible connections in
        the distance matrix.
        """
        i, j = numpy.unravel_index(numpy.nanargmin(clusters), clusters.shape)
        return clusters[i,j], i, j

    def cluster(self, verbose=0, sum_ess=False):
        """
        Cluster all clusters hierarchically until the level of
        num_clusters is obtained. 
        """
        if sum_ess and self.linkage.__name__ != "ward_link":
            raise ValueError(
                "Summing for method other than Ward makes no sense...")
        clusters = copy.copy(self._dist_matrix)
        summed_ess = 0.0

        while len(clusters) > max(self._num_clusters, 1):
            if verbose >= 1:
                print 'k=%s' % len(clusters)
                if verbose == 2:
                    print clusters
            
            best, i, j = self.smallest_distance(clusters)
            # In Ward (1963) ess is summed at each iteration
            # in R's hclust and Python's hcluster and some text books it is not. 
            # Here it is optional...
            if sum_ess:
                summed_ess += best
            else:
                summed_ess = best
            clusters = self.update_distmatrix(i, j, clusters)
            self._dendrogram.merge(i,j)
            self._dendrogram._items[i].distance = summed_ess
            clusters = clusters.remove(j)

    def update_distmatrix(self, i, j, clusters):
        """
        Update the distance matrix using the specified linkage method so that
        it represents the correct distances to the newly formed cluster.
        """
        return self.linkage(clusters, i, j, self._dendrogram)

    def dendrogram(self):
        """Return the dendrogram object."""
        return self._dendrogram

    def num_clusters(self):
        return self._num_clusters

    def __repr__(self):
        return """<Hierarchical Agglomerative Clusterer(linkage method: %r,
                  n=%d clusters>""" % (self.linkage.__name__, self._num_clusters)


class VNClusterer(Clusterer):
    def __init__(self, data, dist_metric=euclidean_distance,
                 linkage=ward_link, num_clusters=1):
        Clusterer.__init__(self, data, dist_metric, linkage, num_clusters)

    def iterate_clusters(self, clusters):
        for i in xrange(1, len(clusters)):
            yield i-1,i

    def smallest_distance(self, clusters):
        best = None
        for i, j in self.iterate_clusters(clusters):
            if best is None or clusters[i][j] <= best[0]:
                best = (clusters[i][j], i, j)
        return best

    def cluster(self, verbose=False):
        # we must sum the error sum of squares in order not to obtain
        # singleton clustering.
        Clusterer.cluster(self, verbose=verbose, sum_ess=True)


def demo():
    """Demo to show some basic functionality."""
    # input vector with two dimensions
    vectors = [[2,4], [0,1], [1,1], [3,2], [4,0], [2,2]]
    # compute the distance matrix on the basis of the vectors
    dist_matrix = DistanceMatrix(vectors, lambda u,v: numpy.sum((u-v)**2)/2)
    # plot the distance matrix
    dist_matrix.draw()
    # initialize a clusterer, with default linkage methode (Ward)
    clusterer = Clusterer(dist_matrix)
    # start the clustering procedure
    clusterer.cluster(verbose=2)
    # plot the result as a dendrogram
    clusterer.dendrogram().draw(title=clusterer.linkage.__name__)


if __name__ == '__main__':
    # TODO argparse options to make this executable.
    demo()
