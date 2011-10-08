# Hierarchical Agglomerative Cluster Analysis
#
# Copyright (C) 2011 Institute for Dutch Lexicology (INL)
# Author: Folgert Karsdorp <folgert.karsdorp@inl.nl>
# URL: <http://www.inl.nl/>
# For licence information, see LICENCE.TXT

def ngrams(text, n):
    """Return N grams of a text."""
    count = max(0, len(text) - n + 1)
    return (tuple(text[i:i+n]) for i in xrange(count))
