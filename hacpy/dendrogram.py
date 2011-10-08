# Hierarchical Agglomerative Cluster Analysis
#
# Copyright (C) 2011 Institute for Dutch Lexicology (INL)
# Author: Folgert Karsdorp <folgert.karsdorp@inl.nl>
# URL: <http://www.inl.nl/>
# For licence information, see LICENCE.TXT

import copy

import numpy
from operator import itemgetter


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
            from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
        except ImportError:
            raise ImportError("Scipy not installed, can't draw dendrogram")
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

